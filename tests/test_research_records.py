"""Current evidence, coding decisions and researcher-controlled definitions."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import GenerationAgent
from codebook import exact_merge, revise_definition, semantic_merge
from research_fixtures import code, response
from research_records import AnalyticDecision, apply_decisions, decisions_for_model


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


if __name__ == "__main__":
    unittest.main()
