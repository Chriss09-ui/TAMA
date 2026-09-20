import sys
import time
import unittest
from pathlib import Path
from threading import Event


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis_job import AnalysisJob


class AnalysisJobTests(unittest.TestCase):
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
