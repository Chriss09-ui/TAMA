"""Run an analysis off the Streamlit script thread with pause checkpoints."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from traceback import extract_tb
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class JobSnapshot:
    stage: str
    pause_requested: bool
    paused: bool
    done: bool
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

    def checkpoint(self, stage: str) -> None:
        with self._lock:
            self._stage = stage
            self._paused = not self._resume.is_set()
        self._resume.wait()
        with self._lock:
            self._paused = False

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(
                stage=self._stage,
                pause_requested=not self._resume.is_set(),
                paused=self._paused,
                done=self.done.is_set(),
                result=self._result,
                error_type=self._error_type,
                error_location=self._error_location,
                http_status=self._http_status,
            )

    def _run(self) -> None:
        try:
            result = self._runner(self.checkpoint)
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
