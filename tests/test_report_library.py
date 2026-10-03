import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis_job import JOB_REGISTRY
from report_library import local_reports, load_library_report


def legacy_report():
    return {"session_name": "older-report", "timestamp": "2026-09-24T11:24:00",
            "generation": {}, "final_themes": [
                {"name": "协作方式", "description": "协作依赖交流。", "codes": ["主动沟通"]}]}


class ReportLibraryTests(unittest.TestCase):
    def setUp(self):
        JOB_REGISTRY.clear_completed()
        for target, value in (("keyring.get_password", None), ("report_library.local_reports", [])):
            mock = patch(target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_discovery_sorts_reports_and_skips_invalid_files(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name, payload in (("old", legacy_report()), ("new", {
                **legacy_report(), "session_name": "newer-report", "timestamp": "2026-10-03T12:00:00"}),
                ("broken", [])):
                folder = directory / name
                folder.mkdir()
                (folder / "00_final_results.json").write_text(json.dumps(payload), encoding="utf-8")
            (directory / "external").symlink_to(directory / "old", target_is_directory=True)
            # The imported reference is intentionally unpatched for this unit check.
            entries = local_reports(directory)
        self.assertEqual([item["session_name"] for item in entries], ["newer-report", "older-report"])
        self.assertEqual(entries[1]["label"], "访谈分析 · 09/24 11:24")

    def test_legacy_report_is_read_only_and_excludes_unverified_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            path.write_text(json.dumps({**legacy_report(), "accepted": True, "metadata": {"final_average_score": 5}}))
            result = load_library_report(path)
        self.assertTrue(result["legacy_read_only"])
        self.assertNotIn("accepted", result)
        self.assertNotIn("metadata", result)
        self.assertNotIn("generation", result)

    def test_modern_report_keeps_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            path.write_text(json.dumps({"session_name": "modern", "codes": [], "final_themes": []}))
            result = load_library_report(path)
            self.assertTrue(result["restored_report"])
            self.assertNotIn("legacy_read_only", result)
            path.write_text(json.dumps({"session_name": "modern", "codes": {}, "final_themes": []}))
            with self.assertRaises(ValueError):
                load_library_report(path)

    def test_navigation_opens_legacy_report_without_editing_tools(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "report.json"
            path.write_text(json.dumps(legacy_report()))
            entry = {"session_name": "older-report", "label": "访谈分析 · 09/24 11:24",
                     "timestamp": "2026-09-24", "path": path}
            with patch("report_library.local_reports", return_value=[entry]):
                app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
                app.button(key="library_item_0").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["analysis_result"]["legacy_read_only"])
        self.assertIn("访谈分析报告", [item.value for item in app.title])
        self.assertTrue(any("旧版报告" in item.value for item in app.info))
        self.assertNotIn("研究复核", [item.label for item in app.tabs])

    def test_new_analysis_keeps_report_and_model_settings_for_return_visit(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
            result = {"session_name": "session-report", "metadata": {}, "codes": [],
                      "final_themes": [], "refinement_iterations": 0}
            app.session_state["analysis_result"] = result
            app.session_state["transcript_text"] = "上次材料"
            app.session_state["provider"] = "MiMo"
            app.run()
            app.button(key="new_analysis").click().run()
            self.assertNotIn("analysis_result", app.session_state)
            self.assertEqual(app.text_area(key="transcript_text").value, "")
            self.assertEqual(app.selectbox(key="provider").value, "MiMo")
            app.button(key="library_item_0").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["analysis_result"]["session_name"], "session-report")

    def test_switching_reports_keeps_researcher_edits(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
            first = {"session_name": "first", "metadata": {}, "codes": [], "final_themes": [],
                     "refinement_iterations": 0, "timestamp": "2026-10-03"}
            second = {**first, "session_name": "second", "timestamp": "2026-10-02"}
            app.session_state["analysis_result"] = first
            app.session_state["library_reports"] = {"second": second}
            app.run()
            app.session_state["analysis_result"]["analytic_storyline"] = "研究者保留的修改"
            app.button(key="library_item_1").click().run()
            self.assertEqual(app.session_state["analysis_result"]["session_name"], "second")
            app.button(key="library_item_0").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["analysis_result"]["analytic_storyline"], "研究者保留的修改")

    def test_unreadable_report_does_not_replace_open_report(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "missing.json"
            entry = {"session_name": "missing", "label": "报告不可用", "timestamp": "", "path": path}
            with patch("report_library.local_reports", return_value=[entry]):
                app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
                app.session_state["analysis_result"] = {"session_name": "current", "metadata": {},
                                                        "codes": [], "final_themes": [], "timestamp": "2026-10-03"}
                app.run()
                app.button(key="library_item_1").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["analysis_result"]["session_name"], "current")
        self.assertTrue(any("无法打开" in item.value for item in app.error))
