"""Run an analysis off the Streamlit script thread with pause checkpoints."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from traceback import extract_tb
from typing import Any, Callable, Dict, Optional

from analysis_progress import (
    DEFAULT_CONCURRENCY, MAX_CONCURRENCY, PhaseRecord, ProgressEvent,
    ProgressRecord, normalize_phase, normalize_unit,
)


class AnalysisCancelled(Exception):
    """Raised at a model-request boundary after cooperative cancellation."""


@dataclass(frozen=True)
class JobSnapshot:
    stage: str
    pause_requested: bool
    paused: bool
    done: bool
    cancelled: bool
    result: Optional[Dict[str, Any]]
    error_type: Optional[str]
    error_location: Optional[str]
    http_status: Optional[int]
    elapsed_seconds: float = 0.0
    stage_elapsed_seconds: float = 0.0
    idle_seconds: float = 0.0
    completed: int = 0
    total: int = 0
    active: int = 0
    failed: int = 0
    unit: str = "任务"
    iteration: int = 0
    max_iterations: int = 0
    phase: str = "准备运行"
    concurrency_limit: int = DEFAULT_CONCURRENCY
    dynamic_concurrency: bool = False
    recent_events: tuple[ProgressRecord, ...] = ()
    phase_history: tuple[PhaseRecord, ...] = ()


class AnalysisJob:
    def __init__(
        self, runner: Callable[[Callable[[str], None]], Dict[str, Any]],
        max_workers: int = DEFAULT_CONCURRENCY,
    ):
        self._validate_concurrency(max_workers)
        self._runner = runner
        self._lock = Lock()
        self._resume = Event()
        self._resume.set()
        self.done = Event()
        self._cancel = Event()
        self._stage = "准备运行"
        self._paused = False
        self._result = None
        self._error_type = None
        self._error_location = None
        self._http_status = None
        self._concurrency_limit = max_workers
        self._dynamic_concurrency = False
        self._started_at = None
        self._finished_at = None
        self._phase_started_at = None
        self._last_progress_at = None
        self._phase = "准备运行"
        self._batch_id = ""
        self._total = 0
        self._unit = "任务"
        self._active_items = set()
        self._completed_items = set()
        self._failed_items = set()
        self._iteration = 0
        self._phase_iteration = 0
        self._max_iterations = 0
        self._recent_events = []
        self._phase_history = []
        self._thread = Thread(target=self._run, name="tama-analysis", daemon=True)

    def start(self) -> None:
        with self._lock:
            now = monotonic()
            self._started_at = now
            self._phase_started_at = now
            self._last_progress_at = now
        self._thread.start()

    @staticmethod
    def _validate_concurrency(value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_CONCURRENCY:
            raise ValueError(f"并发请求数必须为 1 到 {MAX_CONCURRENCY} 的整数")

    def set_concurrency(self, value: int) -> None:
        self._validate_concurrency(value)
        with self._lock:
            if self._concurrency_limit != value:
                self._concurrency_limit = value
                if not self.done.is_set():
                    self._record_locked(monotonic(), f"并发上限调整为 {value}", update_idle=False)

    def concurrency(self) -> int:
        with self._lock:
            return self._concurrency_limit

    def enable_live_control(self) -> None:
        """Mark a runner that passes the concurrency getter to its workers."""
        with self._lock:
            self._dynamic_concurrency = True

    def pause(self) -> None:
        with self._lock:
            if not self.done.is_set() and not self._cancel.is_set():
                self._resume.clear()

    def resume(self) -> None:
        with self._lock:
            self._resume.set()
            self._paused = False

    def cancel(self) -> None:
        """Stop before the next request; a request already sent may finish."""
        with self._lock:
            if not self.done.is_set():
                self._cancel.set()
                self._resume.set()

    def checkpoint(self, stage: str, *, abort_event: Optional[Event] = None) -> None:
        with self._lock:
            phase = normalize_phase(stage)
            if phase != self._phase:
                self._begin_phase_locked(phase, monotonic())
            self._stage = stage
            self._paused = not self._resume.is_set()
        try:
            if abort_event is None:
                self._resume.wait()
            else:
                while not self._resume.is_set():
                    if abort_event.is_set():
                        raise AnalysisCancelled()
                    self._resume.wait(0.2)
                if abort_event.is_set():
                    raise AnalysisCancelled()
            if self._cancel.is_set():
                raise AnalysisCancelled()
        finally:
            with self._lock:
                self._paused = False

    def report_progress(self, event: ProgressEvent) -> None:
        """Accept worker completions independently of pause/cancel checkpoints."""
        with self._lock:
            if self.done.is_set():
                return
            now = monotonic()
            if event.kind == "round":
                self._iteration = max(0, event.iteration)
                self._max_iterations = max(0, event.max_iterations)
                self._record_locked(now, f"开始第 {self._iteration}/{self._max_iterations} 轮评估")
            elif event.kind == "phase":
                phase = normalize_phase(event.stage)
                if phase != self._phase or event.batch_id != self._batch_id:
                    self._begin_phase_locked(
                        phase, now, event.batch_id, max(0, event.total), normalize_unit(event.unit),
                    )
                self._stage = event.stage
            elif event.kind in ("started", "completed"):
                if (event.batch_id != self._batch_id or self._total <= 0
                        or not 1 <= event.index <= self._total
                        or event.index in self._completed_items):
                    return
                if event.kind == "started":
                    if event.index not in self._active_items:
                        self._active_items.add(event.index)
                        self._last_progress_at = now
                else:
                    self._active_items.discard(event.index)
                    self._completed_items.add(event.index)
                    if event.failed:
                        self._failed_items.add(event.index)
                    outcome = "未成功" if event.failed else "完成"
                    self._record_locked(
                        now, f"{self._phase} · {self._unit} {event.index} {outcome}"
                        f"（已处理 {len(self._completed_items)}/{self._total}）",
                    )

    def _record_locked(self, now: float, message: str, update_idle: bool = True) -> None:
        elapsed = max(0.0, now - self._started_at) if self._started_at is not None else 0.0
        self._recent_events.append(ProgressRecord(elapsed, message))
        self._recent_events = self._recent_events[-12:]
        if update_idle:
            self._last_progress_at = now

    def _archive_phase_locked(self, now: float, state: str = "complete") -> None:
        if self._phase_started_at is None:
            return
        if state == "complete" and self._total > len(self._completed_items):
            state = "partial"
        self._phase_history.append(PhaseRecord(
            phase=self._phase,
            elapsed_seconds=max(0.0, now - self._phase_started_at),
            completed=len(self._completed_items),
            total=self._total,
            failed=len(self._failed_items),
            unit=self._unit,
            iteration=self._phase_iteration,
            state=state,
        ))

    def _begin_phase_locked(
        self, phase: str, now: float, batch_id: str = "", total: int = 0, unit: str = "任务",
    ) -> None:
        promoting_checkpoint = phase == self._phase and self._total == 0
        if not promoting_checkpoint:
            self._archive_phase_locked(now)
        self._phase = phase
        self._batch_id = batch_id
        self._total = total
        self._unit = unit
        self._active_items.clear()
        self._completed_items.clear()
        self._failed_items.clear()
        if not promoting_checkpoint or self._phase_started_at is None:
            self._phase_started_at = now
            self._phase_iteration = self._iteration
            self._record_locked(now, f"开始{phase}")

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            now = self._finished_at if self._finished_at is not None else monotonic()
            return JobSnapshot(
                stage=self._stage,
                pause_requested=not self._resume.is_set(),
                paused=self._paused,
                done=self.done.is_set(),
                cancelled=self._cancel.is_set(),
                result=self._result,
                error_type=self._error_type,
                error_location=self._error_location,
                http_status=self._http_status,
                elapsed_seconds=max(0.0, now - self._started_at) if self._started_at is not None else 0.0,
                stage_elapsed_seconds=max(0.0, now - self._phase_started_at)
                    if self._phase_started_at is not None else 0.0,
                idle_seconds=max(0.0, now - self._last_progress_at)
                    if self._last_progress_at is not None else 0.0,
                completed=len(self._completed_items),
                total=self._total,
                active=len(self._active_items),
                failed=len(self._failed_items),
                unit=self._unit,
                iteration=self._iteration,
                max_iterations=self._max_iterations,
                phase=self._phase,
                concurrency_limit=self._concurrency_limit,
                dynamic_concurrency=self._dynamic_concurrency,
                recent_events=tuple(self._recent_events),
                phase_history=tuple(self._phase_history),
            )

    def _run(self) -> None:
        try:
            result = self._runner(self.checkpoint)
        except AnalysisCancelled:
            with self._lock:
                self._stage = "已取消"
        except Exception as exc:
            frames = extract_tb(exc.__traceback__)
            status = getattr(exc, "status_code", None)
            with self._lock:
                self._error_type = type(exc).__name__
                if frames:
                    self._error_location = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}"
                if isinstance(status, int) and 100 <= status <= 599:
                    self._http_status = status
        else:
            with self._lock:
                self._result = result
        finally:
            with self._lock:
                # A final in-flight call may return without another checkpoint.
                # Once cancellation is accepted, it must remain the outcome.
                if self._cancel.is_set():
                    self._stage = "已取消"
                    self._result = None
                    self._error_type = None
                    self._error_location = None
                    self._http_status = None
                now = monotonic()
                self._finished_at = now
                state = "cancelled" if self._cancel.is_set() else "error" if self._error_type else "complete"
                self._archive_phase_locked(now, state)
                self._active_items.clear()
                message = "分析已取消" if self._cancel.is_set() else "分析未完成" if self._error_type else "分析已完成"
                self._record_locked(now, message)
                self._runner = None
                self.done.set()


class JobRegistry:
    """Process-wide latest job, so a browser refresh can reconnect to it."""

    def __init__(self):
        self._lock = Lock()
        self._job: Optional[AnalysisJob] = None

    def current(self) -> Optional[AnalysisJob]:
        with self._lock:
            return self._job

    def start(
        self, runner: Callable[[Callable[[str], None]], Dict[str, Any]],
        max_workers: int = DEFAULT_CONCURRENCY,
    ) -> AnalysisJob:
        with self._lock:
            if self._job is not None and not self._job.done.is_set():
                raise RuntimeError("已有分析正在运行")
            job = AnalysisJob(runner, max_workers=max_workers)
            self._job = job
            job.start()
            return job

    def clear_completed(self) -> None:
        """Forget a completed job and its in-memory transcript-derived result."""
        with self._lock:
            if self._job is not None and not self._job.done.is_set():
                raise RuntimeError("不能清除正在运行的分析")
            self._job = None


JOB_REGISTRY = JobRegistry()
