"""Bounded parallel work with ordered results and completion-based progress."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from threading import Event, Lock, local
from typing import Callable, Iterable, Iterator, Optional, TypeVar, cast
from uuid import uuid4

from analysis_progress import MAX_CONCURRENCY, ProgressEvent
from analysis_job import AnalysisCancelled, AnalysisJob


T = TypeVar("T")
R = TypeVar("R")
_BATCH_STATE = local()


def admission_checkpoint(callback: Optional[Callable[[str], None]]) -> Optional[Callable[[str], None]]:
    """Reuse a job's pause gate without adding calls to ordinary observers."""
    owner = getattr(callback, "__self__", None)
    if isinstance(owner, AnalysisJob) and getattr(callback, "__func__", None) is AnalysisJob.checkpoint:
        return callback
    return None


def check_model_call(callback: Optional[Callable[[str], None]], stage: str) -> None:
    """Gate a request on both user controls and its current batch's outcome."""
    abort = getattr(_BATCH_STATE, "abort", None)
    if abort is not None and abort.is_set():
        raise AnalysisCancelled()
    if callback:
        if admission_checkpoint(callback):
            callback(stage, abort_event=abort)
        else:
            callback(stage)
    if abort is not None and abort.is_set():
        raise AnalysisCancelled()


@contextmanager
def tracked_task(
    stage: str,
    on_progress: Optional[Callable[[ProgressEvent], None]],
    before_task: Optional[Callable[[str], None]] = None,
) -> Iterator[None]:
    """Report a single serial operation without exposing its input or output."""
    batch_id = uuid4().hex
    if on_progress:
        on_progress(ProgressEvent("phase", stage, batch_id=batch_id, total=1))
    try:
        if before_task:
            check_model_call(before_task, stage)
        if on_progress:
            on_progress(ProgressEvent("started", stage, batch_id=batch_id, total=1, index=1))
        yield
    except AnalysisCancelled:
        raise
    except Exception:
        if on_progress:
            on_progress(ProgressEvent("completed", stage, batch_id=batch_id, total=1, index=1, failed=True))
        raise
    else:
        if on_progress:
            on_progress(ProgressEvent("completed", stage, batch_id=batch_id, total=1, index=1))


def ordered_parallel_map(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int,
    stage: str,
    unit: str = "任务",
    concurrency_limit: Optional[Callable[[], int]] = None,
    on_progress: Optional[Callable[[ProgressEvent], None]] = None,
    failed_result: Optional[Callable[[R], bool]] = None,
    before_task: Optional[Callable[[str], None]] = None,
) -> list[R]:
    """Apply current concurrency to queued work, preserving the input order.

    Only active work is submitted. Lowering the limit leaves running tasks alone;
    raising it can dispatch queued tasks even while an earlier task is waiting.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    values = list(items)
    total = len(values)
    batch_id = uuid4().hex
    abort = Event()
    failure_lock = Lock()
    failures: list[BaseException] = []

    def stop_batch(exc: BaseException) -> None:
        with failure_lock:
            if not failures or isinstance(failures[0], AnalysisCancelled) and not isinstance(exc, AnalysisCancelled):
                failures[:] = [exc]
        abort.set()

    def raise_failure() -> None:
        with failure_lock:
            exc = failures[0] if failures else AnalysisCancelled()
        raise exc

    def emit(kind: str, index: int = 0, failed: bool = False) -> None:
        if on_progress:
            on_progress(ProgressEvent(kind, stage, batch_id=batch_id, total=total,
                                      unit=unit, index=index, failed=failed))

    emit("phase")

    def run(index: int) -> R:
        previous_abort = getattr(_BATCH_STATE, "abort", None)
        _BATCH_STATE.abort = abort
        try:
            check_model_call(before_task, stage)
            emit("started", index + 1)
            result = function(values[index])
            failed = bool(failed_result and failed_result(result))
        except AnalysisCancelled as exc:
            stop_batch(exc)
            raise
        except BaseException as exc:
            stop_batch(exc)
            emit("completed", index + 1, failed=True)
            raise
        else:
            emit("completed", index + 1, failed=failed)
            return result
        finally:
            _BATCH_STATE.abort = previous_abort

    if not values:
        return []
    if concurrency_limit is None and max_workers == 1:
        return [run(index) for index in range(total)]

    capacity = min(total, max(max_workers, MAX_CONCURRENCY) if concurrency_limit else max_workers)

    def current_limit() -> int:
        requested = concurrency_limit() if concurrency_limit else max_workers
        return max(1, min(int(requested), capacity))

    results: list[Optional[R]] = [None] * total
    pending: dict[Future[R], int] = {}
    next_index = 0
    with ThreadPoolExecutor(max_workers=capacity) as executor:
        try:
            while next_index < total or pending:
                if abort.is_set():
                    raise_failure()
                while next_index < total and len(pending) < current_limit() and not abort.is_set():
                    pending[executor.submit(run, next_index)] = next_index
                    next_index += 1
                finished, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in finished:
                    results[pending.pop(future)] = future.result()
        except BaseException as exc:
            stop_batch(exc)
            for future in pending:
                future.cancel()
            raise_failure()
    return cast(list[R], results)
