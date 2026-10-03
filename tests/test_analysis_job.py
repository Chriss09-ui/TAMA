import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis_job import AnalysisCancelled, AnalysisJob, JobRegistry
from analysis_progress import DEFAULT_CONCURRENCY, MAX_CONCURRENCY, ProgressEvent


class AnalysisJobTests(unittest.TestCase):
    def test_progress_counts_out_of_order_completions(self):
        job = AnalysisJob(lambda checkpoint: {})
        job.report_progress(ProgressEvent("phase", "提取编码", "chunks", 3, "片段"))
        for index in (1, 2, 3):
            job.report_progress(ProgressEvent("started", "提取编码", "chunks", index=index))
        job.report_progress(ProgressEvent("completed", "提取编码", "chunks", index=3))
        snapshot = job.snapshot()
        self.assertEqual((snapshot.completed, snapshot.total, snapshot.active), (1, 3, 2))
        self.assertEqual(snapshot.phase, "提取编码")
        self.assertIn("片段 3 完成", snapshot.recent_events[-1].message)
        job.report_progress(ProgressEvent("completed", "提取编码", "chunks", index=1, failed=True))
        self.assertEqual((job.snapshot().completed, job.snapshot().failed), (2, 1))

    def test_progress_is_thread_safe_and_duplicate_items_are_ignored(self):
        job = AnalysisJob(lambda checkpoint: {})
        job.report_progress(ProgressEvent("phase", "全文回查", "review", 48, "片段"))

        def finish(index):
            job.report_progress(ProgressEvent("started", "全文回查", "review", index=index))
            event = ProgressEvent("completed", "全文回查", "review", index=index, failed=index % 4 == 0)
            job.report_progress(event)
            job.report_progress(event)
            job.report_progress(ProgressEvent("started", "全文回查", "review", index=index))

        with ThreadPoolExecutor(max_workers=12) as executor:
            list(executor.map(finish, range(1, 49)))
        snapshot = job.snapshot()
        self.assertEqual((snapshot.completed, snapshot.total, snapshot.active, snapshot.failed), (48, 48, 0, 12))
        self.assertLessEqual(len(snapshot.recent_events), 12)

    def test_stale_batch_cannot_change_current_progress(self):
        job = AnalysisJob(lambda checkpoint: {})
        job.report_progress(ProgressEvent("phase", "归纳主题", "first", 3, "组"))
        job.report_progress(ProgressEvent("started", "归纳主题", "first", index=1))
        job.report_progress(ProgressEvent("phase", "归纳主题", "second", 2, "组"))
        job.report_progress(ProgressEvent("completed", "归纳主题", "first", index=1))
        job.report_progress(ProgressEvent("started", "归纳主题", "first", index=2))
        for index in (0, 3):
            job.report_progress(ProgressEvent("completed", "归纳主题", "second", index=index))
        snapshot = job.snapshot()
        self.assertEqual((snapshot.completed, snapshot.total, snapshot.active), (0, 2, 0))
        self.assertEqual(snapshot.phase_history[-1].state, "partial")

    def test_checkpoint_feedback_keeps_current_evaluation_batch(self):
        job = AnalysisJob(lambda checkpoint: {})
        job.report_progress(ProgressEvent("phase", "评估主题", "round-one", 2, "主题"))
        job.report_progress(ProgressEvent("started", "评估主题", "round-one", index=1))
        job.checkpoint("生成评估反馈 · material-derived-name")
        job.checkpoint("完整评估 · material-derived-name")
        job.report_progress(ProgressEvent("completed", "评估主题", "round-one", index=1))
        snapshot = job.snapshot()
        self.assertEqual((snapshot.phase, snapshot.completed, snapshot.total), ("评估主题", 1, 2))
        self.assertNotIn("material-derived-name", repr(snapshot.recent_events))
        self.assertNotIn("material-derived-name", repr(snapshot.phase_history))

    def test_batch_promotes_same_checkpoint_without_duplicate_history(self):
        job = AnalysisJob(lambda checkpoint: {})
        job.checkpoint("归并编码")
        before = job.snapshot()
        job.report_progress(ProgressEvent("phase", "归并编码", "merge", 1))
        after = job.snapshot()
        self.assertEqual(after.phase_history, before.phase_history)
        self.assertEqual(after.recent_events, before.recent_events)
        self.assertEqual(after.total, 1)

    def test_round_updates_do_not_mislabel_previous_phase_history(self):
        job = AnalysisJob(lambda checkpoint: {})
        job.checkpoint("归纳主题")
        job.report_progress(ProgressEvent("round", "评估主题", iteration=1, max_iterations=5))
        job.report_progress(ProgressEvent("phase", "评估主题", "evaluation", 1, "主题"))
        snapshot = job.snapshot()
        self.assertEqual((snapshot.iteration, snapshot.max_iterations), (1, 5))
        self.assertEqual(snapshot.phase_history[-1].iteration, 0)
        job.report_progress(ProgressEvent("completed", "评估主题", "evaluation", index=1))
        job.checkpoint("修订主题")
        self.assertEqual(job.snapshot().phase_history[-1].iteration, 1)
        self.assertEqual(job.snapshot().phase_history[-1].state, "complete")

    def test_elapsed_times_freeze_on_completion_and_record_serial_phases(self):
        entered, release = Event(), Event()

        def run(checkpoint):
            checkpoint("归并编码")
            entered.set()
            release.wait(5)
            return {"accepted": True}

        with patch("analysis_job.monotonic", return_value=100.0) as clock:
            job = AnalysisJob(run)
            job.start()
            try:
                self.assertTrue(entered.wait(2))
                clock.return_value = 105.0
                snapshot = job.snapshot()
                self.assertEqual((snapshot.elapsed_seconds, snapshot.stage_elapsed_seconds, snapshot.idle_seconds), (5.0, 5.0, 5.0))
                clock.return_value = 110.0
                job.checkpoint("映射编码")
                self.assertEqual(job.snapshot().phase_history[-1].elapsed_seconds, 10.0)
                self.assertEqual(job.snapshot().phase_history[-1].state, "complete")
                clock.return_value = 112.0
                release.set()
                self.assertTrue(job.done.wait(2))
                completed = job.snapshot()
                clock.return_value = 1000.0
                frozen = job.snapshot()
                self.assertEqual((frozen.elapsed_seconds, frozen.stage_elapsed_seconds, frozen.idle_seconds), (12.0, 2.0, 0.0))
                self.assertEqual(frozen, completed)
                self.assertEqual(frozen.phase_history[-1].elapsed_seconds, 2.0)
                job.report_progress(ProgressEvent("phase", "全文回查", "late", 3, "片段"))
                self.assertEqual(job.snapshot(), frozen)
            finally:
                release.set()
                self.assertTrue(job.done.wait(2))

    def test_concurrency_control_validates_values_and_registry_initial_limit(self):
        job = AnalysisJob(lambda checkpoint: {})
        self.assertEqual(job.concurrency(), DEFAULT_CONCURRENCY)
        job.set_concurrency(MAX_CONCURRENCY)
        self.assertEqual(job.snapshot().concurrency_limit, MAX_CONCURRENCY)
        for value in (0, MAX_CONCURRENCY + 1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "并发请求数"):
                job.set_concurrency(value)
        self.assertEqual(job.concurrency(), MAX_CONCURRENCY)
        registry = JobRegistry()
        initial = registry.start(lambda checkpoint: {}, max_workers=6)
        self.assertTrue(initial.done.wait(2))
        self.assertEqual(initial.concurrency(), 6)
        with self.assertRaises(ValueError):
            AnalysisJob(lambda checkpoint: {}, max_workers=0)

    def test_live_control_is_only_enabled_explicitly_by_runner(self):
        job = AnalysisJob(lambda checkpoint: {})
        self.assertFalse(job.snapshot().dynamic_concurrency)
        job.set_concurrency(12)
        self.assertFalse(job.snapshot().dynamic_concurrency)
        job.enable_live_control()
        self.assertTrue(job.snapshot().dynamic_concurrency)

    def test_batch_abort_exits_paused_checkpoint_without_resume(self):
        entered, release, abort, request_sent = Event(), Event(), Event(), Event()

        def run(checkpoint):
            entered.set()
            release.wait(5)
            try:
                checkpoint("评估主题", abort_event=abort)
            except AnalysisCancelled:
                # The parallel batch retains its first actual request error.
                raise RuntimeError("peer request failed") from None
            request_sent.set()
            return {"accepted": True}

        job = AnalysisJob(run)
        job.start()
        try:
            self.assertTrue(entered.wait(2))
            job.pause()
            release.set()
            deadline = time.monotonic() + 2
            while not job.snapshot().paused and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(job.snapshot().paused)
            abort.set()
            self.assertTrue(job.done.wait(1))
            snapshot = job.snapshot()
            self.assertFalse(request_sent.is_set())
            self.assertFalse(snapshot.cancelled)
            self.assertFalse(snapshot.paused)
            self.assertEqual(snapshot.error_type, "RuntimeError")
            self.assertIsNone(snapshot.result)
        finally:
            abort.set()
            release.set()
            self.assertTrue(job.done.wait(2))

    def test_aborted_batch_does_not_pass_an_open_checkpoint(self):
        job = AnalysisJob(lambda checkpoint: {})
        abort = Event()
        abort.set()
        with self.assertRaises(AnalysisCancelled):
            job.checkpoint("提取编码", abort_event=abort)
        self.assertFalse(job.snapshot().paused)
        self.assertFalse(job.snapshot().cancelled)

    def test_cancelled_batch_keeps_processed_counts_without_accepting_result(self):
        entered, release = Event(), Event()

        def run(checkpoint):
            job.report_progress(ProgressEvent("phase", "提取编码", "chunks", 2, "片段"))
            job.report_progress(ProgressEvent("started", "提取编码", "chunks", index=1))
            checkpoint("提取编码 · 片段 1")
            entered.set()
            release.wait(5)
            job.report_progress(ProgressEvent("completed", "提取编码", "chunks", index=1))
            checkpoint("提取编码 · 片段 2")
            return {"accepted": True}

        job = AnalysisJob(run)
        job.start()
        try:
            self.assertTrue(entered.wait(2))
            job.pause()
            job.cancel()
            release.set()
            self.assertTrue(job.done.wait(2))
            snapshot = job.snapshot()
            self.assertTrue(snapshot.cancelled)
            self.assertIsNone(snapshot.result)
            self.assertEqual((snapshot.completed, snapshot.total, snapshot.active), (1, 2, 0))
            self.assertEqual(snapshot.phase_history[-1].state, "cancelled")
        finally:
            release.set()
            self.assertTrue(job.done.wait(2))

    def test_registry_reconnects_and_rejects_duplicate_run(self):
        entered = Event()
        release = Event()
        registry = JobRegistry()

        def run(checkpoint):
            checkpoint("提取编码")
            entered.set()
            self.assertTrue(release.wait(2))
            return {"accepted": True}

        job = registry.start(run)
        self.assertTrue(entered.wait(2))
        self.assertIs(registry.current(), job)
        with self.assertRaisesRegex(RuntimeError, "已有分析"):
            registry.start(run)
        release.set()
        self.assertTrue(job.done.wait(2))
        self.assertEqual(registry.current().snapshot().result, {"accepted": True})
        registry.clear_completed()
        self.assertIsNone(registry.current())

    def test_cancel_unblocks_pause_and_prevents_next_request(self):
        first_started = Event()
        finish_first = Event()
        second_started = Event()

        def run(checkpoint):
            checkpoint("第一次")
            first_started.set()
            self.assertTrue(finish_first.wait(2))
            checkpoint("第二次")
            second_started.set()
            return {}

        job = AnalysisJob(run)
        job.start()
        self.assertTrue(first_started.wait(2))
        job.pause()
        job.cancel()
        finish_first.set()
        self.assertTrue(job.done.wait(2))
        self.assertFalse(second_started.is_set())
        self.assertTrue(job.snapshot().cancelled)
        self.assertIsNone(job.snapshot().error_type)

    def test_pause_waits_for_in_flight_call_then_resumes_at_next_checkpoint(self):
        first_started = Event()
        finish_first = Event()
        second_started = Event()

        def run(checkpoint):
            checkpoint("提取编码")
            first_started.set()
            self.assertTrue(finish_first.wait(2))
            checkpoint("归纳主题")
            second_started.set()
            return {"accepted": True}

        job = AnalysisJob(run)
        job.start()
        self.assertTrue(first_started.wait(2))
        job.pause()
        self.assertTrue(job.snapshot().pause_requested)
        finish_first.set()

        deadline = time.monotonic() + 2
        while not job.snapshot().paused and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(job.snapshot().paused)
        self.assertEqual(job.snapshot().stage, "归纳主题")
        self.assertFalse(second_started.is_set())

        job.resume()
        self.assertTrue(job.done.wait(2))
        self.assertTrue(second_started.is_set())
        self.assertEqual(job.snapshot().result, {"accepted": True})

    def test_cancel_during_final_request_discards_late_result(self):
        entered, release = Event(), Event()

        def run(checkpoint):
            checkpoint("最后一次请求")
            entered.set()
            release.wait(5)
            return {"accepted": True}

        job = AnalysisJob(run)
        job.start()
        try:
            self.assertTrue(entered.wait(2))
            job.cancel()
            job.pause()
            self.assertTrue(job.snapshot().cancelled)
            self.assertFalse(job.snapshot().done)
            self.assertFalse(job.snapshot().pause_requested)
        finally:
            release.set()
            self.assertTrue(job.done.wait(2))
        self.assertEqual(job.snapshot().stage, "已取消")
        self.assertIsNone(job.snapshot().result)

    def test_error_from_cancelled_request_does_not_replace_cancellation(self):
        entered, release = Event(), Event()

        def run(checkpoint):
            entered.set()
            release.wait(5)
            raise TimeoutError("request timed out")

        job = AnalysisJob(run)
        job.start()
        try:
            self.assertTrue(entered.wait(2))
            job.cancel()
        finally:
            release.set()
            self.assertTrue(job.done.wait(2))
        self.assertEqual(job.snapshot().stage, "已取消")
        self.assertIsNone(job.snapshot().error_type)
        self.assertIsNone(job.snapshot().error_location)
        self.assertIsNone(job.snapshot().result)

    def test_cancelling_finished_job_keeps_completed_result(self):
        job = AnalysisJob(lambda checkpoint: {"accepted": True})
        job.start()
        self.assertTrue(job.done.wait(2))
        job.cancel()
        self.assertFalse(job.snapshot().cancelled)
        self.assertEqual(job.snapshot().result, {"accepted": True})

    def test_failure_exposes_only_exception_type(self):
        def run(checkpoint):
            raise ValueError("secret-api-key")

        job = AnalysisJob(run)
        job.start()
        self.assertTrue(job.done.wait(2))
        snapshot = job.snapshot()
        self.assertEqual(snapshot.error_type, "ValueError")
        self.assertIn("test_analysis_job.py:", snapshot.error_location)
        self.assertIsNone(snapshot.result)
        self.assertNotIn("secret-api-key", repr(snapshot))

    def test_failure_keeps_only_safe_http_status(self):
        class RequestError(Exception):
            status_code = 401

        def run(checkpoint):
            checkpoint("提取编码")
            raise RequestError("secret-api-key")

        job = AnalysisJob(run)
        job.start()
        self.assertTrue(job.done.wait(2))
        snapshot = job.snapshot()
        self.assertEqual(snapshot.stage, "提取编码")
        self.assertEqual(snapshot.http_status, 401)
        self.assertNotIn("secret-api-key", repr(snapshot))


if __name__ == "__main__":
    unittest.main()
