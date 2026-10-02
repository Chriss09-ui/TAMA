"""Synthetic comparisons, annotations, context and review; no real model calls."""

from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from docx import Document
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.generation_agent import Code
from agents.evaluation_agent import EvaluationAgent
from analysis_job import JOB_REGISTRY
from analysis_records import ensure_theme_ids, interpretations_by_theme, restore_report, update_annotations
from evidence import source_document, validate_evidence
from evidence_views import build_comparison_matrix, evidence_context
from reporting import render_summary_text
from research_records import ResearchWorkspace, SourceMetadata, continuation_context, prepare_previous_result
from research_ui import save_result_records
from streamlit_app import build_result_docx
from tama import TAMAFramework


def result_fixture():
    entries = [
        ("S1", "P1", "C1", "早期", "访谈者：这些原始资料能共享吗？" + "说明背景。" * 90 + "\n受访者：这个不行。", ["这个不行"]),
        ("S2", "P1", "C1", "中期", "访谈者：后来能共享吗？\n受访者：后来就改了，愿意共享。", ["后来就改了"]),
        ("S3", "P1", "C1", "后期", "访谈者：现在呢？\n受访者：仍然愿意共享。", ["仍然愿意共享"]),
        ("S4", "P2", "C2", "", "访谈者：条件呢？\n受访者：我愿意分享摘要，但是拒绝公开原稿。", ["愿意分享摘要", "拒绝公开原稿"]),
        ("S5", "P3", "", "", "受访者：先保留文件。", ["保留文件"]),
    ]
    codes, documents, sources = [], [], []
    for source_id, participant, case, time, body, excerpts in entries:
        source = SourceMetadata(source_id=source_id, participant_id=participant, case_id=case, recorded_at=time)
        sources.append(source)
        documents.append(source_document(source_id, body))
        for excerpt in excerpts:
            start = body.index(excerpt)
            codes.append(Code(code_id=len(codes), name=excerpt, description=excerpt, source_chunks=[0],
                excerpt=excerpt, source_start=start, source_end=start + len(excerpt), context=excerpt,
                speaker="受访者", source=source).model_dump())
    validate_evidence(codes, documents)
    result = {"session_name": "phase2-synthetic", "timestamp": "2026-10-02T00:00:00", "accepted": True,
              "configuration": {"analysis_mode": "thematic", "save_final": False},
              "metadata": {"final_average_score": 4.5}, "refinement_iterations": 1,
              "codes": codes, "source_documents": documents,
              "final_themes": [{"name": "分享取决于资料范围", "description": "比较分享与拒绝的条件", "kind": "pattern",
                                "code_ids": [0, 1, 2, 3, 4], "counterexample_code_ids": [0, 4]},
                               {"name": "保留文件", "description": "保存操作", "kind": "pattern", "code_ids": [5]}],
              "research_workspace": ResearchWorkspace(round_id="phase2-synthetic", sources=sources).model_dump()}
    ensure_theme_ids(result)
    return result


def notes_fixture(result, text="条件与时间共同影响资料分享"):
    return {"theme_notes": [{"theme_id": result["final_themes"][0]["theme_id"], "text": text}],
            "explanations": [
                {"explanation_id": "A", "text": "分享随时间变化", "supporting_code_ids": [1, 2],
                 "contradicting_code_ids": [4], "boundaries": "仅适用于 P1", "unresolved": "无法解释 P2 的条件差异"},
                {"explanation_id": "B", "text": "分享受资料范围限制", "supporting_code_ids": [3],
                 "contradicting_code_ids": [0], "boundaries": "只涉及原稿与摘要", "unresolved": "P1 早期未说明资料范围"}],
            "review": {"status": "reviewed", "researcher": "R1", "reason": "采用范围解释，但保留时间变化的可能", "selected_explanation_id": "B"}}


class ComparisonAndContextTests(unittest.TestCase):
    def test_matrix_keeps_material_participant_and_case_counts_separate(self):
        matrix = build_comparison_matrix(result_fixture())
        self.assertEqual(matrix["counts"]["sources"], 5)
        self.assertEqual(matrix["counts"]["participants"], 3)
        self.assertEqual(matrix["counts"]["cases"], 2)
        states = {row["label"]: row["cells"][0]["state"] for row in matrix["rows"]}
        self.assertEqual(states, {"P1": "混合", "P2": "混合", "P3": "未涉及"})
        time_rows = build_comparison_matrix(result_fixture(), "recorded_at")["rows"]
        self.assertEqual(time_rows[0]["cells"][0]["state"], "反对")
        self.assertEqual(time_rows[1]["cells"][0]["state"], "支持")

    def test_repeated_same_position_and_pending_codes_do_not_inflate_evidence(self):
        result = result_fixture()
        duplicate = {**result["codes"][1], "code_id": 20}
        pending = {**result["codes"][0], "code_id": 21, "evidence_status": "mismatch"}
        result["codes"].extend([duplicate, pending])
        counts = build_comparison_matrix(result)["counts"]
        self.assertEqual(counts["codes"], 7)
        self.assertEqual(counts["evidence"], 6)
        self.assertEqual(counts["participants"], 3)

    def test_unknown_participant_is_not_inferred_from_source_or_speaker(self):
        result = result_fixture()
        for code in result["codes"]:
            code["source"]["participant_id"] = ""
        for source in result["research_workspace"]["sources"]:
            source["participant_id"] = ""
        matrix = build_comparison_matrix(result)
        self.assertEqual(matrix["counts"]["participants"], 0)
        self.assertEqual(len(matrix["rows"]), 5)
        self.assertTrue(all(row["label"].startswith("未知") for row in matrix["rows"]))

    def test_complete_question_survives_a_long_answer_and_does_not_invent_context(self):
        result = result_fixture()
        context = evidence_context(result, result["codes"][0])
        self.assertTrue(context["complete"])
        self.assertIn("这些原始资料能共享吗", context["text"])
        self.assertIn("这个不行", context["text"])
        self.assertNotIn("后来能共享吗", context["text"])
        result["source_documents"] = []
        fallback = evidence_context(result, result["codes"][0])
        self.assertFalse(fallback["complete"])
        self.assertEqual(fallback["text"], "这个不行")

    def test_paragraph_context_uses_actual_saved_body(self):
        result = result_fixture()
        body = "第一段。\n\n第二段里后来就改了，处理方式变化。\n\n第三段。"
        result["source_documents"][1] = source_document("S2", body)
        code = result["codes"][1]
        code["source_start"] = body.index(code["excerpt"])
        code["source_end"] = code["source_start"] + len(code["excerpt"])
        context = evidence_context(result, code)
        self.assertEqual(context["kind"], "所属段落")
        self.assertEqual(context["text"], "第二段里后来就改了，处理方式变化。")


class AnnotationAndReviewTests(unittest.TestCase):
    def test_duplicate_generated_themes_get_distinct_stable_ids(self):
        result = result_fixture()
        first = result["final_themes"][0]
        first.pop("theme_id")
        second = {**deepcopy(first), "description": "另一种解释"}
        third = deepcopy(first)
        result["final_themes"] = [first, second, third]
        ensure_theme_ids(result)
        ids = [theme["theme_id"] for theme in result["final_themes"]]
        self.assertEqual(len(set(ids)), 3)
        saved = update_annotations(result, {"theme_notes": [{"theme_id": ids[1], "text": "第二个主题的诠释"}]})
        restored = restore_report(json.loads(json.dumps(saved)))
        restored["final_themes"].reverse()
        ensure_theme_ids(restored)
        self.assertEqual([theme["theme_id"] for theme in restored["final_themes"]], ids[::-1])
        self.assertEqual(interpretations_by_theme(restored), {ids[1]: "第二个主题的诠释"})

    def test_duplicate_imported_theme_ids_are_rejected(self):
        result = result_fixture()
        result["final_themes"][1]["theme_id"] = result["final_themes"][0]["theme_id"]
        with self.assertRaises(ValueError):
            restore_report(result)

    def test_restore_rejects_invalid_score_history_before_rendering(self):
        invalid = [None, {}, [{"iteration": 1}], [{"iteration": 1, "average_score": None}],
                   [{"iteration": 1, "average_score": "4.5"}], [{"iteration": 1, "average_score": 99}],
                   [{"iteration": 1, "average_score": float("nan")}],
                   [{"iteration": True, "average_score": 4.5}]]
        for history in invalid:
            with self.subTest(history=history):
                result = result_fixture()
                result["score_history"] = history
                with self.assertRaisesRegex(ValueError, "评分历史"):
                    restore_report(result)
        result = result_fixture()
        result["score_history"] = [{"iteration": 1, "average_score": 4.5}]
        self.assertEqual(restore_report(result)["score_history"], result["score_history"])

    def test_notes_survive_json_round_trip_and_theme_reordering(self):
        result = result_fixture()
        saved = update_annotations(result, notes_fixture(result), confirm_review=True)
        restored = restore_report(json.loads(json.dumps(saved)))
        restored["final_themes"].reverse()
        ensure_theme_ids(restored)
        self.assertEqual(interpretations_by_theme(restored)[restored["final_themes"][1]["theme_id"]], "条件与时间共同影响资料分享")
        self.assertNotIn(restored["final_themes"][0]["theme_id"], interpretations_by_theme(restored))
        self.assertEqual(restored["researcher_annotations"]["review"]["status"], "reviewed")
        self.assertFalse(restored["configuration"]["save_final"])
        self.assertNotIn("output_dir", restored)

    def test_note_changes_reset_review_and_keep_before_after_audit(self):
        result = result_fixture()
        saved = update_annotations(result, notes_fixture(result), confirm_review=True)
        changed = deepcopy(saved["researcher_annotations"])
        changed["theme_notes"][0]["text"] = "需缩小为原稿分享的条件"
        updated = update_annotations(saved, changed)
        self.assertEqual(updated["researcher_annotations"]["review"]["status"], "pending")
        self.assertEqual(len(updated["audit_trail"]), 2)
        self.assertEqual(updated["audit_trail"][-1]["details"]["before"]["theme_notes"][0]["text"], "条件与时间共同影响资料分享")

    def test_bad_theme_reference_pending_support_and_unreasoned_review_are_rejected(self):
        result = result_fixture()
        notes = notes_fixture(result)
        notes["theme_notes"][0]["theme_id"] = "missing"
        with self.assertRaises(ValueError):
            update_annotations(result, notes)
        notes = notes_fixture(result)
        result["codes"][1]["evidence_status"] = "mismatch"
        with self.assertRaises(ValueError):
            update_annotations(result, notes)
        notes = {"review": {"status": "reviewed", "researcher": "R1", "reason": ""}}
        with self.assertRaises(ValueError):
            update_annotations(result, notes)

    def test_text_word_and_complete_json_include_saved_notes_and_separate_review(self):
        result = result_fixture()
        saved = update_annotations(result, notes_fixture(result), include_context=True, confirm_review=True)
        text = render_summary_text(saved)
        word = "\n".join(p.text for p in Document(BytesIO(build_result_docx(saved))).paragraphs)
        for report in (text, word):
            for expected in ("条件与时间共同影响资料分享", "研究者已复核", "分享随时间变化", "分享受资料范围限制", "仅适用于 P1", "这些原始资料能共享吗", "分析与复核过程", "混合"):
                self.assertIn(expected, report)
        self.assertEqual(json.loads(json.dumps(saved))["researcher_annotations"]["review"]["selected_explanation_id"], "B")

    def test_private_notes_do_not_enter_python_continuation_context(self):
        result = result_fixture()
        saved = update_annotations(result, notes_fixture(result, "private-interpretation-marker"))
        _, workspace = prepare_previous_result(saved)
        self.assertNotIn("private-interpretation-marker", continuation_context(workspace))

    def test_save_respects_memory_only_and_does_not_mutate_result_on_path_failure(self):
        result = result_fixture()
        candidate = update_annotations(result, notes_fixture(result))
        with tempfile.TemporaryDirectory() as root:
            save_result_records(result, candidate, root)
            self.assertFalse((Path(root) / "outputs").exists())
            original = deepcopy(result)
            candidate = deepcopy(result)
            candidate["configuration"]["save_final"] = True
            candidate["output_dir"] = root
            with self.assertRaises(ValueError):
                save_result_records(result, candidate, root)
            self.assertEqual(result, original)

    def test_restore_rechecks_original_hashes_and_does_not_fabricate_legacy_history(self):
        result = result_fixture()
        result["source_documents"][0]["text"] += "被修改"
        with self.assertRaises(ValueError):
            restore_report(result)
        result = result_fixture()
        restored = restore_report(result)
        self.assertNotIn("audit_trail", restored)
        result["audit_trail"] = [{"bad": "event"}]
        with self.assertRaises(ValueError):
            restore_report(result)

    def test_local_save_refreshes_notes_json_and_text_and_restore_ignores_disk_path(self):
        result = result_fixture()
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "outputs" / result["session_name"]
            folder.mkdir(parents=True)
            result["configuration"]["save_final"] = True
            result["output_dir"] = str(folder)
            candidate = update_annotations(result, notes_fixture(result), confirm_review=True)
            save_result_records(result, candidate, root)
            saved = json.loads((folder / "00_final_results.json").read_text())
            self.assertIn("条件与时间共同影响资料分享", (folder / "00_summary.txt").read_text())
            self.assertEqual(saved["audit_trail"], result["audit_trail"])
            saved["output_dir"] = "/outside/never-write"
            restored = restore_report(saved)
            self.assertNotIn("output_dir", restored)
            self.assertFalse(restored["configuration"]["save_final"])
    def test_original_version_change_invalidates_a_previous_researcher_review(self):
        result = result_fixture()
        saved = update_annotations(result, notes_fixture(result), confirm_review=True)
        saved["source_documents"][0] = source_document("S1", saved["source_documents"][0]["text"] + "\n末尾说明。")
        restored = restore_report(saved)
        self.assertTrue(restored["annotations_need_review"])
        self.assertEqual(restored["researcher_annotations"]["review"]["status"], "pending")


class NonPatternWorkflowTests(unittest.TestCase):
    def test_retiring_last_pattern_does_not_leave_its_score_on_a_counterexample(self):
        fixture = result_fixture()
        themes = [{"name": "候选模式", "description": "需核对", "kind": "pattern", "code_ids": [0]},
                  {"name": "关键反例", "description": "只出现一次", "kind": "counterexample", "code_ids": [0]}]
        with patch("tama.GenerationAgent") as gen, patch("tama.EvaluationAgent") as evaluator, patch("tama.RefinementAgent") as refiner:
            gen.return_value.run.return_value = {"chunks": ["synthetic"], "codes": [fixture["codes"][0]], "themes": themes}
            evaluator.return_value.run.return_value = {"theme_evaluations": [], "average_score": 3.0, "is_acceptable": False, "global_feedback": "需修订"}
            refiner.return_value.run.return_value = {"refined_themes": [], "refinement_plan": {"operations": []}}
            framework = TAMAFramework(api_key="test", corpus_review=False, max_iterations=3)
            result = framework.run_analysis(fixture["source_documents"][0]["text"], source_metadata={"source_id": "S1"}, save_final=False)
        self.assertIsNone(result["metadata"]["final_average_score"])
        self.assertEqual(result["score_history"][0]["average_score"], 3.0)
        self.assertEqual(result["final_themes"][0]["name"], "关键反例")
        self.assertEqual(evaluator.return_value.run.call_count, 1)

    def test_counterexample_is_not_scored_as_a_shared_pattern(self):
        result = result_fixture()
        theme = {"name": "孤立反证", "description": "关键例外", "kind": "counterexample", "code_ids": [0]}
        agent = EvaluationAgent(api_key="test")
        agent.evaluate_theme = Mock(side_effect=AssertionError("must not score a counterexample"))
        evaluation = agent.evaluate_all_themes([theme], result["codes"])
        self.assertIsNone(evaluation.average_score)
        self.assertFalse(evaluation.is_acceptable)

    def test_refinement_receives_patterns_and_preserves_low_frequency_findings(self):
        fixture = result_fixture()
        counter = {"name": "关键反例", "description": "只出现一次", "kind": "counterexample", "code_ids": [0]}
        generation = {"chunks": ["synthetic"], "codes": fixture["codes"],
                      "themes": [fixture["final_themes"][0], counter]}
        evaluation = {"theme_evaluations": [], "average_score": 3.0, "is_acceptable": False, "global_feedback": "需修订"}
        with patch("tama.GenerationAgent") as gen, patch("tama.EvaluationAgent") as evaluator, patch("tama.RefinementAgent") as refiner:
            gen.return_value.run.return_value = generation
            evaluator.return_value.run.side_effect = [evaluation, {**evaluation, "average_score": 4.0, "is_acceptable": True}]
            refiner.return_value.run.return_value = {"refined_themes": [fixture["final_themes"][0]], "refinement_plan": {"operations": []}}
            framework = TAMAFramework(api_key="test", corpus_review=False, max_iterations=2)
            # The compatibility API supplies additional originals; no UI batch input is added.
            previous = deepcopy(fixture)
            previous["codes"] = [code for code in fixture["codes"] if code["source"]["source_id"] != "S1"]
            previous["final_themes"] = []
            previous["source_documents"] = fixture["source_documents"][1:]
            result = framework.run_analysis(fixture["source_documents"][0]["text"], source_metadata={"source_id": "S1"},
                                            previous_result=previous, save_final=False)
        self.assertEqual(len(refiner.return_value.run.call_args.kwargs["themes"]), 1)
        self.assertTrue(any(theme["name"] == "关键反例" for theme in result["final_themes"]))
        self.assertEqual([item["kind"] for item in result["audit_trail"]],
                         ["source_registered", "generation", "corpus_review", "evaluation", "refinement", "evaluation", "analysis_completed"])
        self.assertEqual(result["researcher_annotations"]["review"]["status"], "pending")


class SecondPhaseInterfaceTests(unittest.TestCase):
    def setUp(self):
        JOB_REGISTRY.clear_completed()
        self.vault = patch("keyring.get_password", return_value=None)
        self.vault.start()
        self.addCleanup(self.vault.stop)

    def app(self, result):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.session_state["analysis_result"] = result
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_thematic_mode_has_overall_argument_and_can_save_a_stable_theme_note(self):
        with patch.dict(os.environ, {}, clear=True):
            app = self.app(result_fixture())
            self.assertIsNotNone(app.text_area(key="argument_claim"))
            theme_id = app.session_state["analysis_result"]["final_themes"][0]["theme_id"]
            app.text_area(key=f"theme_interpretation_{theme_id}").set_value("只在当前条件下成立").run()
            app.button(key=f"theme_note_save_{theme_id}").click().run()
            self.assertFalse(app.exception)
            saved = app.session_state["analysis_result"]
            self.assertEqual(interpretations_by_theme(saved)[theme_id], "只在当前条件下成立")
            saved["final_themes"].reverse()
            app.run()
            self.assertEqual(app.text_area(key=f"theme_interpretation_{theme_id}").value, "只在当前条件下成立")
            self.assertIsNone(JOB_REGISTRY.current())

    def test_matrix_links_to_full_context_and_review_requires_reasons(self):
        with patch.dict(os.environ, {}, clear=True):
            app = self.app(result_fixture())
            self.assertTrue(app.dataframe)
            theme_id = app.session_state["analysis_result"]["final_themes"][0]["theme_id"]
            app.checkbox(key=f"matrix_evidence_0_{theme_id}_0").check().run()
            self.assertTrue(any("这些原始资料能共享吗" in item.value for item in app.text))
            app.selectbox(key="explanation_status").set_value("reviewed")
            app.button(key="FormSubmitter:explanation_review_form-保存解释比较与复核").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertIsNone(JOB_REGISTRY.current())

    def test_existing_report_is_opened_locally_without_starting_an_analysis(self):
        source = result_fixture()
        uploaded = BytesIO(json.dumps(update_annotations(source, notes_fixture(source))).encode())
        import streamlit as st
        real_uploader = st.file_uploader
        def uploader(*args, **kwargs):
            return uploaded if kwargs.get("key") == "saved_report_file" else real_uploader(*args, **kwargs)
        with patch.dict(os.environ, {}, clear=True), patch("streamlit.file_uploader", side_effect=uploader), patch("tama.TAMAFramework") as framework:
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
            app.button(key="open_saved_report").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["analysis_result"]["restored_report"])
            self.assertIn("条件与时间共同影响资料分享", render_summary_text(app.session_state["analysis_result"]))
            framework.assert_not_called()
            self.assertIsNone(JOB_REGISTRY.current())

    def test_two_explanations_can_be_saved_and_metadata_correction_resets_review(self):
        with patch.dict(os.environ, {}, clear=True):
            app = self.app(result_fixture())
            for index, text, supports, counters in ((0, "时间变化", [1, 2], [4]), (1, "范围不同", [3], [0])):
                app.text_area(key=f"explanation_{index}_text").set_value(text)
                app.multiselect(key=f"explanation_{index}_supports").set_value(supports)
                app.multiselect(key=f"explanation_{index}_counters").set_value(counters)
            app.selectbox(key="explanation_choice").set_value("解释 2")
            app.selectbox(key="explanation_status").set_value("reviewed")
            app.text_input(key="explanation_researcher").set_value("R1")
            app.text_area(key="explanation_reason").set_value("范围不同更能解释混合情况")
            app.button(key="FormSubmitter:explanation_review_form-保存解释比较与复核").click().run()
            self.assertFalse(app.exception)
            saved = app.session_state["analysis_result"]
            self.assertEqual(len(saved["researcher_annotations"]["explanations"]), 2)
            self.assertEqual(saved["researcher_annotations"]["review"]["status"], "reviewed")
            original = deepcopy(saved["source_documents"])
            app.text_input(key="matrix_group_0_participant_id").set_value("P9")
            app.text_input(key="matrix_group_0_researcher").set_value("R1")
            app.text_area(key="matrix_group_0_reason").set_value("校正已确认的匿名编号")
            app.button(key="FormSubmitter:matrix_group_form_0-保存证据分组").click().run()
            self.assertFalse(app.exception)
            changed = app.session_state["analysis_result"]
            self.assertEqual(changed["codes"][0]["source"]["participant_id"], "P9")
            self.assertEqual(changed["source_documents"], original)
            self.assertEqual(changed["researcher_annotations"]["review"]["status"], "pending")
            self.assertEqual(changed["audit_trail"][-1]["kind"], "evidence_group")
            self.assertIsNone(JOB_REGISTRY.current())


if __name__ == "__main__":
    unittest.main()
