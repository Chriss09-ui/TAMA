import sys
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis_job import AnalysisJob
from parallel_tasks import check_model_call, ordered_parallel_map, tracked_task
from tama import ThreadlineFramework


class ParallelTaskTests(unittest.TestCase):
    def wait_for_paused(self, job):
        deadline = monotonic() + 2
        while not job.snapshot().paused and monotonic() < deadline:
            Event().wait(0.01)
        self.assertTrue(job.snapshot().paused)

    def run_background(self, function):
        outcome = {}

        def run():
            try:
                outcome["result"] = function()
            except BaseException as exc:
                outcome["error"] = exc

        thread = Thread(target=run, daemon=True)
        thread.start()
        return thread, outcome

    def test_completion_progress_does_not_wait_for_the_first_item(self):
        first_started = Event()
        release_first = Event()
        second_completed = Event()
        events = []

        def task(index):
            if index == 0:
                first_started.set()
                if not release_first.wait(3):
                    raise TimeoutError("first item was not released")
            return index

        def progress(event):
            events.append(event)
            if event.kind == "completed" and event.index == 2:
                second_completed.set()

        thread, outcome = self.run_background(lambda: ordered_parallel_map(
            task, range(2), max_workers=2, stage="提取编码", unit="片段", on_progress=progress,
        ))
        try:
            self.assertTrue(first_started.wait(2))
            self.assertTrue(second_completed.wait(2))
            self.assertTrue(thread.is_alive())
            self.assertEqual([event.index for event in events if event.kind == "completed"], [2])
        finally:
            release_first.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome["result"], [0, 1])
        self.assertEqual({event.batch_id for event in events}, {events[0].batch_id})
        self.assertEqual(events[0].total, 2)
        self.assertEqual(events[0].unit, "片段")

    def test_increasing_limit_starts_queued_work_while_first_item_is_waiting(self):
        limit = [1]
        lock = Lock()
        first_started = Event()
        all_started = Event()
        release = Event()
        started = []

        def task(index):
            with lock:
                started.append(index)
                first_started.set()
                if len(started) == 3:
                    all_started.set()
            if not release.wait(3):
                raise TimeoutError("requests were not released")
            return index

        thread, outcome = self.run_background(lambda: ordered_parallel_map(
            task, range(3), max_workers=1, stage="提取编码", concurrency_limit=lambda: limit[0],
        ))
        try:
            self.assertTrue(first_started.wait(2))
            self.assertEqual(started, [0])
            limit[0] = 3
            self.assertTrue(all_started.wait(2))
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome["result"], [0, 1, 2])

    def test_lowering_limit_leaves_active_work_and_limits_next_items(self):
        limit = [3]
        lock = Lock()
        first_three_started = Event()
        next_started = Event()
        release = [Event() for _ in range(3)]
        active = set()
        new_item_active_counts = []

        def task(index):
            with lock:
                active.add(index)
                if len(active) == 3:
                    first_three_started.set()
                if index >= 3:
                    new_item_active_counts.append(len(active))
                    next_started.set()
            try:
                if index < 3 and not release[index].wait(3):
                    raise TimeoutError("active request was not released")
                return index
            finally:
                with lock:
                    active.discard(index)

        thread, outcome = self.run_background(lambda: ordered_parallel_map(
            task, range(6), max_workers=3, stage="提取编码", concurrency_limit=lambda: limit[0],
        ))
        try:
            self.assertTrue(first_three_started.wait(2))
            limit[0] = 1
            release[1].set()
            release[2].set()
            self.assertFalse(next_started.wait(0.35))
            self.assertTrue(thread.is_alive())
        finally:
            for event in release:
                event.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome["result"], list(range(6)))
        self.assertEqual(new_item_active_counts, [1, 1, 1])

    def test_failed_results_are_processed_and_flagged_without_reordering(self):
        events = []
        result = ordered_parallel_map(
            lambda index: (index, index == 1), range(3), max_workers=2,
            stage="全文回查", on_progress=events.append, failed_result=lambda item: item[1],
        )
        completed = [event for event in events if event.kind == "completed"]
        self.assertEqual(result, [(0, False), (1, True), (2, False)])
        self.assertEqual(len(completed), 3)
        self.assertEqual([event.index for event in completed if event.failed], [2])

    def test_exception_reports_failure_and_does_not_start_later_serial_work(self):
        events = []
        started = []

        def task(index):
            started.append(index)
            raise ValueError("synthetic failure")

        with self.assertRaises(ValueError):
            ordered_parallel_map(task, range(3), max_workers=1, stage="提取编码", on_progress=events.append)
        self.assertEqual(started, [0])
        self.assertEqual([(event.index, event.failed) for event in events if event.kind == "completed"], [(1, True)])

    def test_cancel_stops_queued_requests_after_active_requests_end(self):
        active_started = Event()
        release_active = Event()
        lock = Lock()
        started = []

        def run(checkpoint):
            job = checkpoint.__self__

            def request(index):
                checkpoint("提取编码")
                with lock:
                    started.append(index)
                    if len(started) == 2:
                        active_started.set()
                if not release_active.wait(3):
                    raise TimeoutError("active calls did not finish")
                return index

            return {"items": ordered_parallel_map(
                request, range(6), max_workers=2, stage="提取编码",
                concurrency_limit=job.concurrency, on_progress=job.report_progress,
            )}

        job = AnalysisJob(run, max_workers=2)
        job.start()
        try:
            self.assertTrue(active_started.wait(2))
            job.cancel()
            release_active.set()
            self.assertTrue(job.done.wait(3))
        finally:
            release_active.set()
            job.resume()
        self.assertEqual(started, [0, 1])
        self.assertTrue(job.snapshot().cancelled)
        self.assertIsNone(job.snapshot().result)
        self.assertEqual(job.snapshot().failed, 0)

    def test_paused_queued_item_is_not_counted_as_active_work(self):
        first_started = Event()
        release_first = Event()
        calls = []

        def run(checkpoint):
            job = checkpoint.__self__

            def request(index):
                calls.append(index)
                if index == 0:
                    first_started.set()
                    if not release_first.wait(3):
                        raise TimeoutError("active call did not finish")
                return index

            return {"items": ordered_parallel_map(
                request, range(2), max_workers=1, stage="提取编码",
                before_task=checkpoint, concurrency_limit=job.concurrency,
                on_progress=job.report_progress,
            )}

        job = AnalysisJob(run, max_workers=1)
        job.start()
        try:
            self.assertTrue(first_started.wait(2))
            job.pause()
            release_first.set()
            self.wait_for_paused(job)
            self.assertEqual(job.snapshot().active, 0)
            self.assertEqual(job.snapshot().completed, 1)
            self.assertEqual(calls, [0])
            job.resume()
            self.assertTrue(job.done.wait(3))
        finally:
            release_first.set()
            job.cancel()
            job.resume()
        self.assertEqual(calls, [0, 1])

    def test_first_failure_unblocks_paused_sibling_and_preserves_original_error(self):
        first_started = Event()
        release_failure = Event()
        calls = []

        def run(checkpoint):
            job = checkpoint.__self__

            def request(index):
                calls.append(index)
                if index == 0:
                    first_started.set()
                    if not release_failure.wait(3):
                        raise TimeoutError("failure was not released")
                    raise ValueError("synthetic failure")
                return index

            return {"items": ordered_parallel_map(
                request, range(2), max_workers=1, stage="评估主题",
                before_task=checkpoint, concurrency_limit=job.concurrency,
                on_progress=job.report_progress,
            )}

        job = AnalysisJob(run, max_workers=1)
        job.start()
        try:
            self.assertTrue(first_started.wait(2))
            job.pause()
            job.set_concurrency(2)
            self.wait_for_paused(job)
            release_failure.set()
            self.assertTrue(job.done.wait(3))
            snapshot = job.snapshot()
            self.assertEqual(snapshot.error_type, "ValueError")
            self.assertFalse(snapshot.cancelled)
            self.assertEqual(snapshot.failed, 1)
            self.assertEqual(calls, [0])
        finally:
            release_failure.set()
            job.cancel()
            job.resume()

    def test_batch_failure_prevents_active_siblings_next_model_request(self):
        first_two_started = Event()
        failure_reported = Event()
        lock = Lock()
        first_requests = []
        later_requests = []

        def run(checkpoint):
            job = checkpoint.__self__

            def request(index):
                check_model_call(checkpoint, "评估主题")
                with lock:
                    first_requests.append(index)
                    if len(first_requests) == 2:
                        first_two_started.set()
                if not first_two_started.wait(3):
                    raise TimeoutError("sibling did not start")
                if index == 0:
                    raise ValueError("synthetic failure")
                if not failure_reported.wait(3):
                    raise TimeoutError("failure was not reported")
                check_model_call(checkpoint, "生成评估反馈")
                later_requests.append(index)
                return index

            def progress(event):
                job.report_progress(event)
                if event.kind == "completed" and event.failed:
                    failure_reported.set()

            return {"items": ordered_parallel_map(
                request, range(2), max_workers=2, stage="评估主题",
                before_task=checkpoint, on_progress=progress,
            )}

        job = AnalysisJob(run, max_workers=2)
        job.start()
        try:
            self.assertTrue(job.done.wait(3))
        finally:
            first_two_started.set()
            failure_reported.set()
            job.cancel()
            job.resume()
        self.assertEqual(set(first_requests), {0, 1})
        self.assertEqual(later_requests, [])
        self.assertEqual(job.snapshot().error_type, "ValueError")

    def test_tracked_serial_operation_starts_after_its_checkpoint(self):
        events = []

        def checkpoint(stage):
            self.assertEqual(stage, "归并编码")
            self.assertEqual([event.kind for event in events], ["phase"])

        with tracked_task("归并编码", events.append, checkpoint):
            self.assertEqual([event.kind for event in events], ["phase", "started"])
        self.assertEqual([event.kind for event in events], ["phase", "started", "completed"])
        self.assertFalse(events[-1].failed)

    def test_framework_wires_progress_control_and_saves_measured_duration(self):
        code = {"code_id": 0, "description": "记录", "excerpt": "记录",
                "source_start": 0, "source_end": 2, "source": {"source_id": "synthetic"}}
        generation = {"chunks": ["合成片段"], "codes": [code],
                      "themes": [{"name": "记录", "description": "合成主题", "code_ids": [0]}]}
        evaluation = {"theme_evaluations": [], "average_score": 4.5,
                      "is_acceptable": True, "global_feedback": "合成反馈"}
        events = []
        on_progress = events.append
        concurrency_limit = lambda: 2

        with tempfile.TemporaryDirectory() as output_dir, \
                patch("tama.GenerationAgent") as generation_cls, \
                patch("tama.EvaluationAgent") as evaluation_cls, \
                patch("tama.RefinementAgent"), \
                patch("tama.monotonic", side_effect=[10.0, 15.25]), redirect_stdout(io.StringIO()):
            generation_cls.return_value.run.return_value = generation
            evaluation_cls.return_value.run.return_value = evaluation
            framework = ThreadlineFramework(api_key="offline-test", output_dir=output_dir, corpus_review=False)
            result = framework.run_analysis(
                "记录", session_name="duration-test", source_metadata={"source_id": "synthetic"},
                on_progress=on_progress, concurrency_limit=concurrency_limit,
            )
            saved = json.loads((Path(output_dir) / "duration-test" / "00_final_results.json").read_text())

        self.assertEqual(result["metadata"]["elapsed_seconds"], 5.25)
        self.assertEqual(saved["metadata"]["elapsed_seconds"], 5.25)
        self.assertIs(generation_cls.return_value.on_progress, on_progress)
        self.assertIs(evaluation_cls.return_value.concurrency_limit, concurrency_limit)
        self.assertEqual([(event.kind, event.iteration, event.max_iterations) for event in events], [("round", 1, 5)])


if __name__ == "__main__":
    unittest.main()
