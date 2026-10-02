import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import Chunk, Code, GenerationAgent
from evidence_fixtures import matched_code
from agents.refinement_agent import RefinementAgent
from decisions.base import DecisionAnswer


def fake_response(data):
    message = SimpleNamespace(content=json.dumps(data))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class GenericPromptTests(unittest.TestCase):
    def assert_generic_prompt(self, client):
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        prompt = "\n".join(message["content"] for message in messages).lower()
        for domain_term in ("clinical", "aao", "cardiac", "parent", "patient"):
            self.assertNotIn(domain_term, prompt)
        self.assertIn("质性研究者", messages[0]["content"])
        self.assertIn("相同的语言", prompt)
        self.assertIn("只返回符合以下结构的 json 对象", prompt)

    def test_generation_prompts_are_grounded_and_domain_neutral(self):
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        stages = []
        agent.before_model_call = stages.append
        agent.client.chat.completions.create.return_value = fake_response(
            {"codes": [{"description": "通勤安排的变化"}]}
        )
        agent.generate_codes_from_chunk(
            Chunk(chunk_id=0, text="受访者描述了通勤安排的变化。", start_word=0, end_word=1)
        )
        self.assertEqual(stages, ["提取编码 · 片段 1"])
        self.assert_generic_prompt(agent.client)
        code_prompt = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn('"codes"', code_prompt)
        self.assertIn('"description"', code_prompt)

        agent.client.chat.completions.create.return_value = fake_response(
            {"themes": [{"name": "日常安排", "description": "日常出行的调整。", "codes": ["通勤安排的变化"]}]}
        )
        agent.generate_themes([matched_code(code_id=0, description="通勤安排的变化", source_chunks=[0])])
        self.assertEqual(stages[-1], "归纳主题")
        self.assert_generic_prompt(agent.client)
        theme_prompt = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn('"themes"', theme_prompt)
        self.assertIn('"name"', theme_prompt)

    def test_evaluation_prompt_uses_generic_criteria(self):
        # Hybrid evaluation scores via the decision client, then the mocked
        # main model writes feedback; the generic-prompt assertions apply to
        # that feedback call.
        with patch("agents.evaluation_agent.LLMDecisionClient") as decision_client_cls:
            provider = decision_client_cls.return_value
            provider.ask.return_value = {
                "coverage": DecisionAnswer(key="coverage", kind="score", score=2.0, confidence=0.9),
                "actionability": DecisionAnswer(key="actionability", kind="score", score=3.0, confidence=0.9),
                "distinctiveness": DecisionAnswer(key="distinctiveness", kind="score", score=3.0, confidence=0.9),
                "relevance": DecisionAnswer(key="relevance", kind="score", score=3.0, confidence=0.9),
                "needs_refinement": DecisionAnswer(
                    key="needs_refinement", kind="noul", probability=0.4, confidence=0.8,
                ),
            }
            agent = EvaluationAgent(api_key="test")
            agent.client = Mock()
            stages = []
            agent.before_model_call = stages.append
            agent.client.chat.completions.create.return_value = fake_response({
                "coverage_feedback": "涵盖主要编码",
                "actionability_feedback": "概念明确",
                "distinctiveness_feedback": "区别清晰",
                "relevance_feedback": "有编码支持",
                "needs_refinement": False, "refinement_suggestions": []
            })
            theme = {"name": "日常安排", "description": "日常出行的调整。", "codes": ["通勤安排的变化"]}
            agent.evaluate_theme(theme, [theme], [matched_code(code_id=0, description="通勤安排的变化").model_dump()])
        self.assertEqual(stages, ["评估主题 · 日常安排", "生成评估反馈 · 日常安排"])
        self.assert_generic_prompt(agent.client)
        self.assertIn("追溯到原文", agent.criteria.relevance)
        feedback_prompt = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn('"coverage_feedback"', feedback_prompt)
        self.assertIn('"needs_refinement"', feedback_prompt)

    def test_refinement_prompt_stays_grounded_and_uses_source_language(self):
        agent = RefinementAgent(api_key="test")
        agent.client = Mock()
        stages = []
        agent.before_model_call = stages.append
        agent.client.chat.completions.create.return_value = fake_response(
            {"operations": [], "summary": "无需调整"}
        )
        agent.create_refinement_plan(
            themes=[{"name": "日常安排", "description": "日常出行的调整。", "codes": []}],
            evaluation_results={"theme_evaluations": [], "global_feedback": "已达标"},
            codes=[{"description": "通勤安排的变化"}],
        )
        self.assertEqual(stages, ["修订主题"])
        self.assert_generic_prompt(agent.client)
        refinement_prompt = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn('"operation": "add"', refinement_prompt)
        self.assertIn("日常安排", refinement_prompt)
        self.assertNotIn("\\u65e5", refinement_prompt)

    def test_global_feedback_passed_to_refinement_is_chinese(self):
        agent = EvaluationAgent(api_key="test")
        evaluation = SimpleNamespace(
            needs_refinement=True,
            coverage_score=3,
            actionability_score=4,
            distinctiveness_score=4,
            relevance_score=4,
        )
        feedback = agent._generate_global_feedback([evaluation], 3.75, False)
        self.assertIn("需要修订", feedback)
        self.assertIn("覆盖度", feedback)
        self.assertNotIn("NEEDS REFINEMENT", feedback)

    def test_code_quote_is_located_and_invalid_quote_requires_review(self):
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        chunk = Chunk(
            chunk_id=2, text="受访者：合同在法务部门。", start_word=0, end_word=1,
            start_char=40, end_char=53,
        )
        agent.client.chat.completions.create.return_value = fake_response({"codes": [
            {
                "description": "合同由法务掌握", "excerpt": "合同在法务部门",
                "focus": ["evidence_holder"], "statement_type": "participant_report",
                "verification_status": "reported_only", "open_question": "",
            },
            {
                "description": "公开资料待核查", "excerpt": "官网已有合同",
                "focus": ["public_status"], "statement_type": "participant_report",
                "verification_status": "supported_by_material", "open_question": "",
            },
        ]})

        codes = agent.generate_codes_from_chunk(chunk)
        self.assertEqual(codes[0].source_start, 44)
        self.assertEqual(codes[0].source_end, 51)
        self.assertEqual(codes[0].excerpt, chunk.text[4:11])
        self.assertEqual(codes[0].verification_status, "reported_only")
        self.assertIsNone(codes[1].excerpt)
        self.assertIsNone(codes[1].source_start)
        self.assertEqual(codes[1].verification_status, "unknown")
        self.assertIn("人工核对", codes[1].open_question)

        agent.client.chat.completions.create.return_value = fake_response({
            "codes": [{"description": "缺少来源的编码"}],
        })
        missing_quote = agent.generate_codes_from_chunk(chunk)[0]
        self.assertIn("缺少逐字原文", missing_quote.open_question)

    def test_theme_keeps_code_references_and_rejects_unsourced_mechanism(self):
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        agent.client.chat.completions.create.return_value = fake_response({"themes": [
            {
                "name": "部门间断点", "description": "证据未流入市场部门。",
                "kind": "information_breakpoint", "code_ids": [3, 999],
                "counterexample_code_ids": [], "open_questions": [],
                "codes": ["法务掌握合同"],
            },
            {
                "name": "无来源解释", "description": "待核查。",
                "kind": "candidate_mechanism", "code_ids": [999],
                "counterexample_code_ids": [], "open_questions": [], "codes": [],
            },
        ]})
        themes = agent.generate_themes([
            matched_code(code_id=3, description="法务掌握合同", source_chunks=[0]),
        ])

        self.assertEqual(themes[0].code_ids, [3])
        self.assertEqual(themes[0].codes, ["法务掌握合同"])
        self.assertEqual(themes[1].kind, "evidence_gap")
        self.assertIn("缺少可追溯", themes[1].open_questions[0])


if __name__ == "__main__":
    unittest.main()
