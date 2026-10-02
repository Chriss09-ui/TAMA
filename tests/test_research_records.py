import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.evaluation_agent import EvaluationAgent
from evidence_fixtures import matched_code
from evidence import source_document
from agents.generation_agent import Code, GenerationAgent
from agents.refinement_agent import RefinementAgent, RefinementOperation, RefinementPlan
from codebook import exact_merge, revise_definition, semantic_merge
from next_data_plan import build_next_data_plan
from reporting import render_summary_text
from research_records import (
    AnalyticDecision, ArgumentRecord, CategoryRecord, ComparisonRecord, ResearchWorkspace,
    SamplingTask, SourceMetadata, apply_decisions, build_workspace, continuation_context,
    decisions_for_model, prepare_previous_result, validate_workspace,
)
from tama import TAMAFramework


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])


def code(code_id=0, excerpt="在内部保留资料", source_id="S1", **kwargs):
    return matched_code(code_id=code_id, name="控制信息", description="控制信息", source_chunks=[0],
                excerpt=excerpt, source_start=0, source_end=len(excerpt),
                source=SourceMetadata(source_id=source_id), **kwargs)


class EvidenceAndDecisionTests(unittest.TestCase):
    def test_label_match_never_merges_distinct_evidence(self):
        codes = [code(), code(1, "决定向客户分享资料", source_id="S2")]
        exact_merge(codes)
        self.assertTrue(all(item.merged_into is None for item in codes))

    def test_identical_evidence_deduplicates_but_protected_evidence_does_not(self):
        codes = [code(), code(1)]
        exact_merge(codes)
        self.assertEqual(codes[1].merged_into, 0)
        self.assertEqual(codes[0].merged_from, [1])
        other = [code(separate_from=[1]), code(1)]
        exact_merge(other)
        self.assertIsNone(other[1].merged_into)
        locked = [code(definition_locked=True, definition="研究者已修改"), code(1)]
        exact_merge(locked)
        self.assertIsNone(locked[1].merged_into)

    def test_definition_change_keeps_previous_boundary_and_requires_review(self):
        original = code(definition="内部存储", include="研发", exclude="客户")
        revise_definition(original, {"definition": "控制共享权限", "exclude": "仅存储"}, "两种行动有区别", "R1")
        self.assertEqual(original.definition_history[0]["before"]["definition"], "内部存储")
        self.assertEqual(original.definition_history[0]["after"]["exclude"], "仅存储")
        self.assertEqual(original.definition_review, "需复核")

    def test_unfit_evidence_stays_out_of_the_proposed_merge(self):
        codes = [code(), code(1, "只是保留备份")]
        semantic_merge(codes, [{"name": "设置共享权限", "definition": "决定谁可使用", "code_ids": [0, 1],
                               "evidence_reviews": [{"code_id": 1, "status": "does_not_fit", "reason": "存储不等于权限"}]}])
        self.assertIsNone(codes[1].merged_into)
        self.assertEqual(codes[1].review_note, "存储不等于权限")

    def test_confirmed_separation_reopens_the_complete_family(self):
        codes = [code(), code(1), code(2)]
        semantic_merge(codes, [{"code_ids": [0, 1, 2], "definition": "旧定义"}])
        decision = AnalyticDecision(action="保持分开", text="行动不同", reason="对照三条证据", code_ids=[0, 1],
                                    researcher="R1", confirmed=True)
        apply_decisions(codes, [decision.model_dump()])
        self.assertTrue(all(item.merged_into is None for item in codes))
        self.assertTrue(all(not item.merged_from for item in codes))
        with self.assertRaisesRegex(ValueError, "保持分开"):
            semantic_merge(codes, [{"code_ids": [0, 1]}])

    def test_researcher_definition_is_locked_against_model_overwrite(self):
        codes = [code()]
        decision = AnalyticDecision(action="修改定义", text="研究者新定义", reason="比较旧原话", code_ids=[0], researcher="R1", confirmed=True)
        apply_decisions(codes, [decision.model_dump()])
        semantic_merge(codes, [{"code_ids": [0], "definition": "模型替换定义"}])
        self.assertEqual(codes[0].definition, "研究者新定义")

    def test_only_explicitly_confirmed_decisions_enter_the_model_view(self):
        decision = {"action": "否决解释", "text": "资料不足以说明能力弱", "reason": "只能看到公开材料", "researcher": "R1",
                    "human_note": "私人反思", "confirmed": False}
        self.assertEqual(decisions_for_model([decision]), ())
        decision["confirmed"] = True
        visible = str(decisions_for_model([decision]))
        self.assertIn("能力弱", visible)
        self.assertNotIn("私人反思", visible)
        with self.assertRaises(ValueError):
            decisions_for_model([{**decision, "confirmed": "true"}])

    def test_every_family_excerpt_and_context_reaches_merge_prompt(self):
        codes = [code(context="访谈者：你给谁看？\n受访者：在内部保留资料"), code(1, "只让审核人员看")]
        semantic_merge(codes, [{"code_ids": [0, 1], "definition": "控制资料权限"}])
        codes.append(code(2, "客户拿到全部资料", source_id="S2"))
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        agent.client.chat.completions.create.return_value = response({"groups": []})
        agent.consolidate_codebook(codes)
        prompt = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        for expected in ("你给谁看", "只让审核人员看", "客户拿到全部资料", "S2"):
            self.assertIn(expected, prompt)

    def test_counterexample_context_reaches_the_evaluator(self):
        codes = [code().model_dump(), code(1, "对外全部提供", context="这是一次反例", source_id="S2").model_dump()]
        packed = EvaluationAgent._compact_codes({"code_ids": [0], "counterexample_code_ids": [1]}, codes)
        self.assertEqual(packed[1]["context"], "这是一次反例")
        self.assertEqual(packed[1]["source"]["source_id"], "S2")


class ContinuingStudyTests(unittest.TestCase):
    def test_import_rejects_duplicate_ids_and_merge_cycles(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            prepare_previous_result({"codes": [code().model_dump(), code().model_dump()]})
        codes = [code().model_dump(), code(1).model_dump()]
        codes[0]["merged_into"], codes[1]["merged_into"] = 1, 0
        with self.assertRaisesRegex(ValueError, "循环"):
            prepare_previous_result({"codes": codes})

    def test_import_rejects_a_family_with_a_wrong_reverse_reference(self):
        codes = [code().model_dump(), code(1).model_dump()]
        codes[0]["merged_from"] = [1]
        with self.assertRaisesRegex(ValueError, "前后不一致"):
            prepare_previous_result({"codes": codes})

    def test_redefining_a_merged_child_reopens_its_evidence(self):
        codes = [code(), code(1)]
        semantic_merge(codes, [{"code_ids": [0, 1], "definition": "旧定义"}])
        decision = AnalyticDecision(action="修改定义", text="只适用于第二条的新定义", reason="行动不同", code_ids=[1],
                                    researcher="R1", confirmed=True)
        apply_decisions(codes, [decision.model_dump()])
        self.assertIsNone(codes[1].merged_into)
        self.assertEqual(codes[1].definition, "只适用于第二条的新定义")
        self.assertTrue(codes[1].definition_locked)
        self.assertEqual(codes[0].merged_from, [])

    def test_legacy_result_gains_stable_evidence_identity_without_private_fields(self):
        result = {"session_name": "old", "codes": [{"code_id": 7, "description": "旧编码", "human_note": "私密"}],
                  "memos": [{"human_note": "不提交"}], "reflexivity": "隐私"}
        codes, workspace = prepare_previous_result(result)
        self.assertEqual(codes[0]["evidence_id"], "old:7")
        self.assertEqual(codes[0]["source"]["source_id"], "old")
        self.assertNotIn("私密", json.dumps(codes, ensure_ascii=False))
        self.assertNotIn("隐私", continuation_context(workspace))

    def test_unknown_references_are_rejected_instead_of_silently_disappearing(self):
        workspace = ResearchWorkspace(focused_code_ids=[99])
        with self.assertRaisesRegex(ValueError, "不存在的编码"):
            validate_workspace(workspace, [code().model_dump()])

    def test_new_data_reopens_judgements_and_keeps_prior_history(self):
        category = CategoryRecord(name="权限", code_ids=[0], status="研究者判断饱和",
                                  judgement="比较多个不同事件后暂时判断", researcher="R1")
        previous = ResearchWorkspace(round_id="old", rounds=[{"round_id": "old"}], categories=[category])
        mapping = {"iterations": [{"step": 2, "groups": [{"name": "权限", "code_ids": [0, 1]}]}]}
        result = build_workspace([code().model_dump(), code(1, source_id="new").model_dump()], mapping,
                                 "next", SourceMetadata(source_id="new"), previous)
        self.assertEqual(result["categories"][0]["status"], "待发展")
        self.assertEqual(result["categories"][0]["history"][0]["status"], "研究者判断饱和")
        self.assertEqual(previous.categories[0].status, "研究者判断饱和")
        self.assertEqual(result["round_number"], 2)

    def test_sampling_distinguishes_fact_checks_from_theoretical_purpose(self):
        base = dict(gap="资料不能区分权限与表达", target="不同事件", rationale="观察不同条件", researcher="R1")
        with self.assertRaises(ValueError):
            SamplingTask(kind="理论抽样", **base)
        task = SamplingTask(kind="理论抽样", category_id="c1", expected_change="区分两种解释", **base)
        plan = build_next_data_plan([{"code_id": 0, "open_question": "询问负责人"}], [], [],
                                    {"sampling_tasks": [task.model_dump()]})
        self.assertEqual(plan["tasks"][0], task.model_dump())
        self.assertNotIn("区分两种解释", plan["markdown"])
        self.assertNotIn("观察不同条件", plan["markdown"])
        with self.assertRaises(ValueError):
            SamplingTask(status="已分析", **base)

    def test_comparison_and_argument_require_real_evidence_and_author(self):
        with self.assertRaises(ValueError):
            ComparisonRecord(code_ids=[0, 0], differences="不同", implication="拆分", researcher="R1")
        with self.assertRaises(ValueError):
            ArgumentRecord(claim="已有结论", researcher="R1")
        with self.assertRaises(ValueError):
            CategoryRecord(name="类别", status="研究者判断饱和")

    def test_framework_continues_old_ids_and_does_not_use_score_as_saturation(self):
        old = code(4).model_dump()
        previous_workspace = ResearchWorkspace(round_id="old", rounds=[{"round_id": "old"}],
            categories=[CategoryRecord(name="权限", code_ids=[4], properties="旧属性", researcher="R1")])
        previous = {"session_name": "old", "codes": [old], "research_workspace": previous_workspace.model_dump(),
                    "memos": [{"human_note": "不应进入模型"}], "source_documents": [source_document("S1", old["excerpt"])]}
        with tempfile.TemporaryDirectory() as temp, patch("tama.EvaluationAgent") as evaluator:
            evaluator.return_value.run.return_value = {"theme_evaluations": [], "average_score": 5,
                "is_acceptable": True, "global_feedback": "主题评估通过"}
            framework = TAMAFramework(api_key="test", output_dir=temp, analysis_mode="grounded_theory", corpus_review=False)
            framework.generation_agent.client = Mock()
            framework.generation_agent.client.chat.completions.create.side_effect = [
                response({"codes": [{"name": "对外共享", "excerpt": "给客户资料", "speaker": "受访者"}]}),
                response({"groups": [{"code_ids": [4]}, {"code_ids": [5]}]}),
                response({"categories": [{"name": "权限", "code_ids": [4, 5]}]}),
                response({"analytic_storyline": "分享方式随条件变化", "themes": [{"name": "权限变化", "description": "条件不同", "code_ids": [4, 5]}]}),
            ]
            result = framework.run_analysis("访谈者：给谁？\n受访者：给客户资料", session_name="new", save_final=False,
                previous_result=previous, source_metadata={"source_id": "S2", "participant_id": "P1"}, include_previous_context=True)
            self.assertEqual([item["code_id"] for item in result["codes"]], [4, 5])
            self.assertEqual(result["codes"][0]["evidence_id"], "old:4")
            self.assertEqual(result["codes"][1]["source"]["participant_id"], "P1")
            self.assertIn("给谁", result["codes"][1]["context"])
            self.assertEqual(result["research_workspace"]["project_id"], previous_workspace.project_id)
            self.assertEqual(result["research_workspace"]["round_number"], 2)
            self.assertEqual(result["research_workspace"]["categories"][0]["status"], "待发展")
            prompts = str(framework.generation_agent.client.chat.completions.create.call_args_list)
            self.assertIn("旧属性", prompts)
            first_prompt = framework.generation_agent.client.chat.completions.create.call_args_list[0].kwargs["messages"][1]["content"]
            self.assertIn("P1", first_prompt)
            self.assertIn("本份材料的来源信息", first_prompt)
            self.assertNotIn("不应进入模型", prompts)
            self.assertFalse(Path(temp).joinpath("new").exists())

    def test_refinement_preserves_counterexamples_through_renaming(self):
        agent = RefinementAgent(api_key="test")
        plan = RefinementPlan(operations=[RefinementOperation(operation="combine", target_themes=["A", "B"],
            rationale="共享模式", new_theme={"name": "C"})], summary="合并")
        result = agent.apply_refinement_plan([
            {"name": "A", "counterexample_code_ids": [9], "open_questions": ["待核查"]},
            {"name": "B", "counterexample_code_ids": [10], "uncertain": "不同解释"},
        ], plan)
        self.assertEqual(result[0]["counterexample_code_ids"], [9, 10])
        self.assertIn("不同解释", result[0]["open_questions"])

    def test_report_does_not_present_obsolete_initial_storyline_as_final(self):
        text = render_summary_text({"accepted": True, "codes": [], "final_themes": [],
            "generation": {"analytic_storyline": "旧的结论"}, "analytic_storyline": "", "storyline_needs_review": True})
        self.assertNotIn("旧的结论", text)
        self.assertIn("重新核对整体论证", text)
        self.assertIn("不代表理论饱和", text)


if __name__ == "__main__":
    unittest.main()
