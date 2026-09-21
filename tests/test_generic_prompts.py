import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import Chunk, Code, GenerationAgent
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
        agent.generate_themes([Code(code_id=0, description="通勤安排的变化", source_chunks=[0])])
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
            agent.evaluate_theme(theme, [theme], [{"description": "通勤安排的变化"}])
        self.assertEqual(stages, ["评估主题 · 日常安排", "生成评估反馈 · 日常安排"])
        self.assert_generic_prompt(agent.client)
        self.assertIn("提供的编码", agent.criteria.relevance)
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


if __name__ == "__main__":
    unittest.main()
