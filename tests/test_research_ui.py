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
from agents.generation_agent import Code
from evidence import source_document, validate_evidence
from research_records import CategoryRecord, ResearchWorkspace, SourceMetadata
from research_ui import save_workspace


def study_result(round_id="R1"):
    codes = [Code(code_id=value, description=f"编码{value}", source_chunks=[0], excerpt=f"原话{value}",
                  context=f"访谈者：问题{value}\n受访者：原话{value}", source=SourceMetadata(source_id=f"S{value}")).model_dump()
             for value in (0, 1)]
    documents = [source_document(f"S{value}", f"原话{value}") for value in (0, 1)]
    for code in codes:
        code.update(source_start=0, source_end=len(code["excerpt"]))
    validate_evidence(codes, documents)
    workspace = ResearchWorkspace(round_id=round_id, rounds=[{"round_id": round_id}],
        categories=[CategoryRecord(category_id="C0", name="资料保留", code_ids=[0]),
                    CategoryRecord(category_id="C1", name="资料分享", code_ids=[1])])
    return {"session_name": round_id, "accepted": True, "codes": codes, "source_documents": documents, "refinement_iterations": 1,
            "configuration": {"analysis_mode": "grounded_theory", "save_final": False},
            "metadata": {"final_average_score": 4.5}, "final_themes": [],
            "research_workspace": workspace.model_dump()}


class ResearchEditorTests(unittest.TestCase):
    def setUp(self):
        JOB_REGISTRY.clear_completed()
        vault = patch("keyring.get_password", return_value=None)
        vault.start()
        self.addCleanup(vault.stop)

    def test_grounded_mode_renders_all_editors_and_saves_a_comparison(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
            app.session_state["analysis_result"] = study_result()
            app.run()
            self.assertFalse(app.exception)
            app.multiselect(key="comparison_code_ids").set_value([0, 1]).run()
            app.text_area(key="comparison_differences").set_value("两条资料的分享条件不同")
            app.text_area(key="comparison_implication").set_value("不能只按名称合并")
            app.text_input(key="comparison_researcher").set_value("R1")
            app.button(key="FormSubmitter:comparison_form-保存比较记录").click().run()
            self.assertFalse(app.exception)
            comparisons = app.session_state["analysis_result"]["research_workspace"]["comparisons"]
            self.assertEqual(comparisons[0]["code_ids"], [0, 1])
            self.assertEqual(comparisons[0]["implication"], "不能只按名称合并")
            self.assertIsNone(JOB_REGISTRY.current())

    def test_insufficient_judgement_does_not_save_and_new_round_clears_old_widgets(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
            app.session_state["analysis_result"] = study_result()
            app.run()
            app.selectbox(key="category_C0_status").set_value("研究者判断饱和")
            app.button(key="FormSubmitter:category_form_C0-保存类属发展记录").click().run()
            self.assertFalse(app.exception)
            self.assertIn("比较依据", app.error[0].value)
            self.assertEqual(app.session_state["analysis_result"]["research_workspace"]["categories"][0]["status"], "待发展")
            app.session_state["analysis_result"] = study_result("R2")
            app.run()
            self.assertEqual(app.selectbox(key="category_C0_status").value, "待发展")

    def test_decision_is_saved_locally_without_model_submission(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
            app.session_state["analysis_result"] = study_result()
            app.run()
            app.multiselect(key="decision_code_ids").set_value([0, 1])
            app.text_area(key="decision_text").set_value("这两条编码保持分开")
            app.text_area(key="decision_reason").set_value("行动与条件不同")
            app.text_input(key="decision_researcher").set_value("R1")
            app.checkbox(key="decision_confirmed").check()
            app.button(key="FormSubmitter:analysis_decision_form-保存分析决定").click().run()
            self.assertFalse(app.exception)
            workspace = app.session_state["analysis_result"]["research_workspace"]
            self.assertTrue(workspace["decisions"][0]["confirmed"])
            self.assertEqual(workspace["submitted_decision_ids"], [])
            checkbox_keys = [control.key for control in app.checkbox]
            self.assertNotIn("continue_current_research", checkbox_keys)
            self.assertFalse(any(control.key == "submitted_analysis_decisions" for control in app.multiselect))
            self.assertIsNone(JOB_REGISTRY.current())

    def test_grounded_mode_and_explicit_source_reach_the_runner(self):
        result = study_result()
        with patch.dict(os.environ, {}, clear=True), patch("tama.TAMAFramework") as framework:
            framework.return_value.run_analysis.return_value = result
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
            app.selectbox(key="analysis_mode").set_value("建构扎根理论支持").run()
            app.text_input(key="source_source_id").set_value("Interview-02").run()
            app.text_area(key="transcript_text").set_value("受访者：新资料。").run()
            app.text_input(key="api_key_DeepSeek").set_value("test").run()
            app.button(key="run_analysis").click().run()
            self.assertTrue(app.session_state["analysis_job"].done.wait(2))
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(framework.call_args.kwargs["analysis_mode"], "grounded_theory")
            self.assertEqual(framework.return_value.run_analysis.call_args.kwargs["source_metadata"]["source_id"], "Interview-02")

    def test_saved_edits_refresh_json_text_and_codebook_in_the_correct_directory(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "outputs" / "R1"
            output.mkdir(parents=True)
            result = study_result()
            result["configuration"]["save_final"] = True
            result["output_dir"] = str(output)
            workspace = ResearchWorkspace.model_validate(result["research_workspace"])
            workspace.argument.claim = "分享与保留须区分"
            workspace.argument.supporting_code_ids = [0, 1]
            workspace.argument.researcher = "R1"
            save_workspace(result, workspace, root)
            saved = json.loads((output / "00_final_results.json").read_text())
            self.assertEqual(saved["research_workspace"]["argument"]["claim"], "分享与保留须区分")
            self.assertIn("分享与保留须区分", (output / "00_summary.txt").read_text())
            self.assertTrue((output / "08_research_records.md").is_file())
            result["output_dir"] = root
            with self.assertRaisesRegex(ValueError, "保存位置"):
                save_workspace(result, workspace, root)
            self.assertFalse((Path(root) / "00_final_results.json").exists())

    def test_single_submission_has_no_continuation_or_sampling_editors(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
            app.session_state["analysis_result"] = study_result()
            app.run()
            self.assertFalse(app.exception)
            keys = [control.key or "" for control in app.selectbox]
            self.assertFalse(any(key.startswith("task_") for key in keys))
            self.assertNotIn("sampling_task_selection", keys)
            self.assertNotIn("continue_current_research", [control.key for control in app.checkbox])
            self.assertNotIn("previous_research_json", [control.key for control in app.get("file_uploader")])

    def test_relationship_graph_renders_only_after_a_supported_relation_is_saved(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
            app.session_state["analysis_result"] = study_result()
            app.run()
            self.assertEqual(len(app.get("graphviz_chart")), 0)
            app.text_area(key="relation_description").set_value("两种行动在不同条件下关联")
            app.text_input(key="relation_researcher").set_value("R1")
            app.button(key="FormSubmitter:category_relationship_form-保存类属关系").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(app.session_state["analysis_result"]["research_workspace"]["argument"]["relationships"], [])
            self.assertEqual(len(app.get("graphviz_chart")), 0)

            app.multiselect(key="relation_codes").set_value([0, 1])
            app.button(key="FormSubmitter:category_relationship_form-保存类属关系").click().run()
            self.assertFalse(app.exception)
            relations = app.session_state["analysis_result"]["research_workspace"]["argument"]["relationships"]
            self.assertEqual(relations[0]["code_ids"], [0, 1])
            self.assertEqual(relations[0]["researcher"], "R1")
            graphs = app.get("graphviz_chart")
            self.assertEqual(len(graphs), 1)
            spec = graphs[0].proto.spec
            self.assertRegex(spec, r'"?C0"?\s*->\s*"?C1"?')
            for label in ("资料保留", "资料分享", relations[0]["description"]):
                self.assertIn(label, spec)
            self.assertIsNone(JOB_REGISTRY.current())

    def test_full_text_review_defaults_on_and_can_be_disabled(self):
        with patch.dict(os.environ, {}, clear=True), patch("tama.TAMAFramework") as framework:
            framework.return_value.run_analysis.return_value = study_result()
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
            self.assertTrue(app.checkbox(key="corpus_review").value)
            app.checkbox(key="corpus_review").uncheck().run()
            app.text_area(key="transcript_text").set_value("受访者：已有的文稿。").run()
            app.text_input(key="api_key_DeepSeek").set_value("test").run()
            app.button(key="run_analysis").click().run()
            self.assertTrue(app.session_state["analysis_job"].done.wait(2))
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(framework.call_args.kwargs["corpus_review"])
            self.assertNotIn("previous_result", framework.return_value.run_analysis.call_args.kwargs)

    def test_unscored_partial_review_renders_pending_codes_and_downloads(self):
        result = study_result()
        result["codes"][0]["evidence_status"] = "unverified"
        result["metadata"]["final_average_score"] = None
        result["accepted"] = False
        result["stop_reason"] = "corpus_review_incomplete"
        result["corpus_review"] = {"status": "partial", "total_chunks": 2, "reviewed_chunks": 1,
                                  "findings": [], "failed_chunks": [{"chunk_id": 1, "error_type": "ValueError"}]}
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
            app.session_state["analysis_result"] = result
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any(metric.value == "未评分" for metric in app.metric))
            self.assertTrue(any("部分完成" in warning.value for warning in app.warning))
            self.assertTrue(any("待核查编码" in expander.label for expander in app.expander))
            self.assertEqual(len(app.get("download_button")), 2)

    def test_independent_result_clears_previous_interpretation_widget(self):
        first = study_result()
        first["final_themes"] = [{"name": "第一份文稿主题", "description": "描述", "code_ids": [0]}]
        second = study_result("R2")
        second["final_themes"] = [{"name": "第二份文稿主题", "description": "描述", "code_ids": [0]}]
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
            app.session_state["analysis_result"] = first
            app.run()
            first_id = app.session_state["analysis_result"]["final_themes"][0]["theme_id"]
            app.text_area(key=f"theme_interpretation_{first_id}").set_value("第一份的人工说明").run()
            app.session_state["analysis_result"] = second
            app.run()
            self.assertFalse(app.exception)
            second_id = app.session_state["analysis_result"]["final_themes"][0]["theme_id"]
            self.assertNotEqual(first_id, second_id)
            self.assertEqual(app.text_area(key=f"theme_interpretation_{second_id}").value, "")



if __name__ == "__main__":
    unittest.main()
