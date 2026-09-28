"""Run an analysis off the Streamlit script thread with pause checkpoints."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from traceback import extract_tb
from typing import Any, Callable, Dict, Optional


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


class AnalysisJob:
    def __init__(self, runner: Callable[[Callable[[str], None]], Dict[str, Any]]):
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
        self._thread = Thread(target=self._run, name="tama-analysis", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if not self.done.is_set():
                self._resume.clear()

    def resume(self) -> None:
        with self._lock:
            self._resume.set()
            self._paused = False

    def cancel(self) -> None:
        """Stop before the next request; a request already sent may finish."""
        self._cancel.set()
        self._resume.set()

    def checkpoint(self, stage: str) -> None:
        with self._lock:
            self._stage = stage
            self._paused = not self._resume.is_set()
        self._resume.wait()
        if self._cancel.is_set():
            raise AnalysisCancelled()
        with self._lock:
            self._paused = False

    def snapshot(self) -> JobSnapshot:
        with self._lock:
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

    def start(self, runner: Callable[[Callable[[str], None]], Dict[str, Any]]) -> AnalysisJob:
        with self._lock:
            if self._job is not None and not self._job.done.is_set():
                raise RuntimeError("已有分析正在运行")
            job = AnalysisJob(runner)
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
