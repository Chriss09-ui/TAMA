import json
import os
import sys
import unittest
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evidence_fixtures import matched_code
from agents.evaluation_agent import EvaluationAgent, EvaluationResult
from decisions.base import DecisionAnswer, DecisionProvider, DecisionProviderError
from decisions.jev_client import JevDecisionClient
from prompts import CRITERION_SCALES


THEME = {"name": "日常安排", "description": "日常出行的调整。", "codes": ["通勤安排的变化"]}
CODES = [matched_code(code_id=0, description="通勤安排的变化").model_dump()]


def fake_response(data):
    message = SimpleNamespace(content=json.dumps(data))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def decision_answers(
    coverage=3.0, actionability=3.0, distinctiveness=3.0, relevance=3.0,
    confidence=0.95, needs_probability=0.1, needs_confidence=0.9,
):
    """Build a full answer map; raw scores are zero-based level indices."""
    values = {
        "coverage": ("score", coverage),
        "actionability": ("score", actionability),
        "distinctiveness": ("score", distinctiveness),
        "relevance": ("score", relevance),
    }
    answers = {
        key: DecisionAnswer(key=key, kind=kind, score=value, confidence=confidence)
        for key, (kind, value) in values.items()
    }
    answers["needs_refinement"] = DecisionAnswer(
        key="needs_refinement", kind="noul",
        probability=needs_probability, confidence=needs_confidence,
    )
    return answers


class StubDecisionProvider(DecisionProvider):
    name = "stub"

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def ask(self, state, questions):
        self.calls.append((state, questions))
        return self.answers


class HybridEvaluationTests(unittest.TestCase):
    def make_agent(self, provider):
        agent = EvaluationAgent(api_key="test", decision_provider=provider)
        agent.client = Mock()
        stages = []
        agent.before_model_call = stages.append
        return agent, stages

    def test_fallback_rejects_scores_outside_the_rating_range(self):
        for score in (0, 99, float("inf"), True):
            with self.subTest(score=score):
                provider = StubDecisionProvider(decision_answers())
                provider.ask = Mock(side_effect=RuntimeError("offline failure"))
                agent, _ = self.make_agent(provider)
                payload = {f"{key}_score": score for key in ("coverage", "actionability", "distinctiveness", "relevance")}
                payload.update({f"{key}_feedback": "反馈" for key in ("coverage", "actionability", "distinctiveness", "relevance")})
                payload.update(needs_refinement=False, refinement_suggestions=[])
                agent.client.chat.completions.create.return_value = fake_response(payload)
                with self.assertRaises(ValueError):
                    agent.evaluate_all_themes([THEME], CODES)

    def test_feedback_parses_false_string_without_requesting_refinement(self):
        provider = StubDecisionProvider(decision_answers(confidence=0.5))
        agent, _ = self.make_agent(provider)
        agent.client.chat.completions.create.return_value = fake_response({
            "coverage_feedback": "反馈", "actionability_feedback": "反馈",
            "distinctiveness_feedback": "反馈", "relevance_feedback": "反馈",
            "needs_refinement": "false", "refinement_suggestions": [],
        })
        result = agent.evaluate_theme(THEME, [THEME], CODES)
        self.assertFalse(result.needs_refinement)
        self.assertTrue(result.flagged_for_review)

    def test_fallback_keeps_integer_valued_float_ratings_compatible(self):
        provider = StubDecisionProvider(decision_answers())
        provider.ask = Mock(side_effect=RuntimeError("offline failure"))
        agent, _ = self.make_agent(provider)
        payload = {f"{key}_score": 4.0 for key in ("coverage", "actionability", "distinctiveness", "relevance")}
        payload.update({f"{key}_feedback": "反馈" for key in ("coverage", "actionability", "distinctiveness", "relevance")})
        payload.update(needs_refinement=False, refinement_suggestions=[])
        agent.client.chat.completions.create.return_value = fake_response(payload)
        result = agent.evaluate_all_themes([THEME], CODES)
        self.assertEqual(result.average_score, 4.0)
        self.assertTrue(result.is_acceptable)


    def test_strong_theme_skips_feedback_call(self):
        provider = StubDecisionProvider(decision_answers())
        agent, stages = self.make_agent(provider)

        result = agent.evaluate_theme(THEME, [THEME], CODES)

        self.assertEqual(len(provider.calls), 1)
        agent.client.chat.completions.create.assert_not_called()
        self.assertEqual(stages, ["评估主题 · 日常安排"])
        self.assertEqual(result.coverage_score, 4)
        self.assertEqual(result.overall_score, 4.0)
        self.assertFalse(result.needs_refinement)
        self.assertEqual(result.refinement_suggestions, [])
        self.assertEqual(result.feedback_source, "placeholder")
        self.assertFalse(result.flagged_for_review)
        self.assertEqual(result.score_confidences["coverage"], 0.95)
        self.assertEqual(result.raw_scores["coverage"], 3.0)

    def test_decision_state_contains_theme_codes_and_context(self):
        provider = StubDecisionProvider(decision_answers())
        agent, _ = self.make_agent(provider)

        agent.evaluate_theme(THEME, [THEME], CODES)

        state, questions = provider.calls[0]
        self.assertEqual(state["theme"]["codes"], ["通勤安排的变化"])
        self.assertEqual(state["original_codes"][0]["excerpt"], CODES[0]["excerpt"])
        self.assertEqual(state["original_codes"][0]["code_id"], 0)
        self.assertEqual(set(questions), {
            "coverage", "actionability", "distinctiveness", "relevance", "needs_refinement",
        })
        self.assertEqual(questions["coverage"].scale, list(CRITERION_SCALES["coverage"]))
        self.assertIn("不得高于 2 分", questions["actionability"].instructions)
        self.assertNotEqual(questions["coverage"].scale, questions["actionability"].scale)
        self.assertEqual(questions["needs_refinement"].kind, "noul")
        self.assertIn("覆盖度", questions["needs_refinement"].instructions)
        self.assertIn("区分度", questions["needs_refinement"].instructions)

    def test_low_score_theme_triggers_feedback_call(self):
        provider = StubDecisionProvider(decision_answers(coverage=2.0))
        agent, stages = self.make_agent(provider)
        agent.client.chat.completions.create.return_value = fake_response({
            "coverage_feedback": "缺少部分编码的覆盖",
            "actionability_feedback": "概念明确",
            "distinctiveness_feedback": "区别清晰",
            "relevance_feedback": "有编码支持",
            "needs_refinement": True,
            "refinement_suggestions": ["补充编码"],
        })

        result = agent.evaluate_theme(THEME, [THEME], CODES)

        self.assertEqual(stages, ["评估主题 · 日常安排", "生成评估反馈 · 日常安排"])
        agent.client.chat.completions.create.assert_called_once()
        self.assertEqual(result.coverage_score, 3)
        self.assertEqual(result.overall_score, 3.75)
        self.assertEqual(result.coverage_feedback, "缺少部分编码的覆盖")
        self.assertTrue(result.needs_refinement)
        self.assertEqual(result.refinement_suggestions, ["补充编码"])
        self.assertEqual(result.feedback_source, "llm")

    def test_low_confidence_flags_review_and_triggers_feedback(self):
        provider = StubDecisionProvider(decision_answers(confidence=0.5))
        agent, stages = self.make_agent(provider)
        agent.client.chat.completions.create.return_value = fake_response({
            "coverage_feedback": "反馈", "actionability_feedback": "反馈",
            "distinctiveness_feedback": "反馈", "relevance_feedback": "反馈",
            "needs_refinement": False, "refinement_suggestions": [],
        })

        result = agent.evaluate_theme(THEME, [THEME], CODES)

        self.assertEqual(stages[-1], "生成评估反馈 · 日常安排")
        self.assertTrue(result.flagged_for_review)

    def test_score_rounding_is_half_up_and_raw_scores_are_kept(self):
        provider = StubDecisionProvider(decision_answers(
            coverage=3.5, actionability=2.5, distinctiveness=0.2, relevance=4.4,
        ))
        agent, _ = self.make_agent(provider)
        agent.client.chat.completions.create.return_value = fake_response({
            "coverage_feedback": "反馈", "actionability_feedback": "反馈",
            "distinctiveness_feedback": "反馈", "relevance_feedback": "反馈",
            "needs_refinement": False, "refinement_suggestions": [],
        })

        result = agent.evaluate_theme(THEME, [THEME], CODES)

        self.assertEqual(result.coverage_score, 5)  # 0-based 3.5 -> half-up -> 5
        self.assertEqual(result.actionability_score, 4)  # 2.5 -> 4
        self.assertEqual(result.distinctiveness_score, 1)  # 0.2 -> 1
        self.assertEqual(result.relevance_score, 5)  # 4.4 -> 5
        self.assertEqual(
            result.raw_scores,
            {"coverage": 3.5, "actionability": 2.5, "distinctiveness": 0.2, "relevance": 4.4},
        )
        self.assertEqual(
            result.weighted_scores,
            {"coverage": 4.5, "actionability": 3.5, "distinctiveness": 1.2, "relevance": 5.0},
        )
        self.assertTrue(result.needs_refinement)

    def test_average_cannot_hide_a_weak_criterion(self):
        agent, _ = self.make_agent(StubDecisionProvider(decision_answers()))

        def evaluation(name, coverage):
            return EvaluationResult(
                theme_name=name,
                coverage_score=round(coverage), coverage_feedback="反馈",
                actionability_score=5, actionability_feedback="反馈",
                distinctiveness_score=5, distinctiveness_feedback="反馈",
                relevance_score=5, relevance_feedback="反馈",
                overall_score=(coverage + 15) / 4,
                needs_refinement=False, refinement_suggestions=[],
                weighted_scores={
                    "coverage": coverage, "actionability": 5.0,
                    "distinctiveness": 5.0, "relevance": 5.0,
                },
            )

        agent.evaluate_theme = Mock(side_effect=[evaluation("主题甲", 5.0), evaluation("主题乙", 3.5)])
        themes = [{"name": "主题甲", "code_ids": [0]}, {"name": "主题乙", "code_ids": [0]}]
        agent.max_workers = 1
        result = agent.evaluate_all_themes(themes, CODES, acceptance_threshold=4.0)

        self.assertGreater(result.average_score, 4.0)
        self.assertFalse(result.is_acceptable)
        self.assertIn("尚未满足逐项验收标准", result.global_feedback)

    def test_continuous_score_below_threshold_triggers_feedback(self):
        provider = StubDecisionProvider(decision_answers(coverage=2.6))
        agent, stages = self.make_agent(provider)
        agent.client.chat.completions.create.return_value = fake_response({
            "coverage_feedback": "覆盖度尚未达到四分",
            "actionability_feedback": "反馈",
            "distinctiveness_feedback": "反馈",
            "relevance_feedback": "反馈",
            "needs_refinement": True,
            "refinement_suggestions": ["补充覆盖"],
        })

        result = agent.evaluate_theme(THEME, [THEME], CODES)

        self.assertEqual(result.coverage_score, 4)
        self.assertAlmostEqual(result.weighted_scores["coverage"], 3.6)
        self.assertAlmostEqual(result.overall_score, 3.9)
        self.assertEqual(stages[-1], "生成评估反馈 · 日常安排")

    def test_needs_refinement_probability_gates_detail_call(self):
        # A prose feedback call cannot cancel a positive typed decision.
        provider = StubDecisionProvider(decision_answers(needs_probability=0.6))
        agent, stages = self.make_agent(provider)
        agent.client.chat.completions.create.return_value = fake_response({
            "coverage_feedback": "反馈", "actionability_feedback": "反馈",
            "distinctiveness_feedback": "反馈", "relevance_feedback": "反馈",
            "needs_refinement": False, "refinement_suggestions": [],
        })

        result = agent.evaluate_theme(THEME, [THEME], CODES)

        self.assertEqual(stages[-1], "生成评估反馈 · 日常安排")
        self.assertTrue(result.needs_refinement)

        # 0.4 -> confident no refinement; strong scores -> placeholder only.
        provider = StubDecisionProvider(decision_answers(needs_probability=0.4))
        agent, stages = self.make_agent(provider)

        result = agent.evaluate_theme(THEME, [THEME], CODES)

        self.assertEqual(stages, ["评估主题 · 日常安排"])
        self.assertFalse(result.needs_refinement)
        self.assertEqual(result.feedback_source, "placeholder")

    def test_recoverable_decision_failure_falls_back_to_full_evaluation(self):
        provider = StubDecisionProvider(decision_answers())
        provider.ask = Mock(side_effect=RuntimeError("network down"))
        agent, stages = self.make_agent(provider)
        agent.client.chat.completions.create.return_value = fake_response({
            "coverage_score": 4, "coverage_feedback": "涵盖主要编码",
            "actionability_score": 4, "actionability_feedback": "概念明确",
            "distinctiveness_score": 4, "distinctiveness_feedback": "区别清晰",
            "relevance_score": 4, "relevance_feedback": "有编码支持",
            "needs_refinement": False, "refinement_suggestions": [],
        })

        result = agent.evaluate_theme(THEME, [THEME], CODES)

        self.assertEqual(stages, ["评估主题 · 日常安排", "完整评估 · 日常安排"])
        agent.client.chat.completions.create.assert_called_once()
        self.assertEqual(result.coverage_score, 4)
        self.assertIsNone(result.score_confidences)
        self.assertEqual(result.feedback_source, "fallback")

    def test_permanent_decision_failure_propagates(self):
        provider = StubDecisionProvider(decision_answers())
        provider.ask = Mock(side_effect=DecisionProviderError("配置错误", permanent=True))
        agent, _ = self.make_agent(provider)

        with self.assertRaises(DecisionProviderError):
            agent.evaluate_theme(THEME, [THEME], CODES)

    def test_jev_mode_sends_five_questions_in_one_request(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx_response()

        def httpx_response():
            import httpx
            return httpx.Response(200, json={
                "answers": {
                    "coverage": {"score": 3.0, "confidence": 0.9},
                    "actionability": {"score": 3.0, "confidence": 0.9},
                    "distinctiveness": {"score": 3.0, "confidence": 0.9},
                    "relevance": {"score": 3.0, "confidence": 0.9},
                    "needs_refinement": {"noul": 0.1},
                },
            })

        provider = JevDecisionClient(
            api_key="jev-test", transport=make_transport(handler),
        )
        agent = EvaluationAgent(api_key="test", decision_provider=provider)

        agent.evaluate_theme(THEME, [THEME], CODES)

        body = json.loads(requests[0].content)
        self.assertEqual(set(body["questions"]), {
            "coverage", "actionability", "distinctiveness", "relevance", "needs_refinement",
        })
        self.assertEqual(len(body["questions"]["coverage"]["criteria"]), 5)

    def test_provider_calls_run_concurrently_with_order_kept(self):
        barrier = Barrier(3)
        themes = [
            {"name": f"主题 {i}", "description": "描述", "codes": [], "code_ids": [0]} for i in range(3)
        ]

        class BlockingProvider(StubDecisionProvider):
            def ask(self, state, questions):
                barrier.wait(timeout=5)
                return decision_answers()

        agent = EvaluationAgent(api_key="test", max_workers=3, decision_provider=BlockingProvider(None))

        result = agent.evaluate_all_themes(themes, CODES)

        self.assertEqual([e.theme_name for e in result.theme_evaluations],
                         ["主题 0", "主题 1", "主题 2"])

    def test_empty_theme_list_has_clear_error(self):
        agent = EvaluationAgent(
            api_key="test", decision_provider=StubDecisionProvider(decision_answers()),
        )

        with self.assertRaisesRegex(ValueError, "没有可评估的主题"):
            agent.evaluate_all_themes([], CODES)

    def test_invalid_confidence_threshold_is_rejected(self):
        for threshold in (0, -0.1, 1.5):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    EvaluationAgent(api_key="test", confidence_threshold=threshold)

    def test_global_feedback_reports_flagged_themes(self):
        agent = EvaluationAgent(api_key="test")
        evaluation = SimpleNamespace(
            needs_refinement=False, coverage_score=4, actionability_score=4,
            distinctiveness_score=4, relevance_score=4, flagged_for_review=True,
        )
        feedback = agent._generate_global_feedback([evaluation], 4.0, True)
        self.assertIn("建议人工复核", feedback)

    def test_run_records_decision_audit_fields(self):
        provider = StubDecisionProvider(decision_answers(confidence=0.5))
        agent = EvaluationAgent(api_key="test", decision_provider=provider)
        agent.client = Mock()
        agent.client.chat.completions.create.return_value = fake_response({
            "coverage_feedback": "反馈", "actionability_feedback": "反馈",
            "distinctiveness_feedback": "反馈", "relevance_feedback": "反馈",
            "needs_refinement": False, "refinement_suggestions": [],
        })

        result = agent.run([THEME], CODES)

        self.assertEqual(result["decision_provider"], "stub")
        self.assertEqual(result["confidence_threshold"], 0.7)
        self.assertEqual(result["flagged_themes"], ["日常安排"])
        saved = result["theme_evaluations"][0]
        self.assertEqual(saved["score_confidences"]["needs_refinement"], 0.9)
        self.assertTrue(saved["flagged_for_review"])

    def test_evaluation_result_defaults_stay_backward_compatible(self):
        result = EvaluationResult(
            theme_name="x", coverage_score=4, coverage_feedback="f",
            actionability_score=4, actionability_feedback="f",
            distinctiveness_score=4, distinctiveness_feedback="f",
            relevance_score=4, relevance_feedback="f",
            overall_score=4.0, needs_refinement=False, refinement_suggestions=[],
        )
        self.assertIsNone(result.score_confidences)
        self.assertIsNone(result.needs_refinement_confidence)
        self.assertIsNone(result.raw_scores)
        self.assertIsNone(result.weighted_scores)
        self.assertFalse(result.flagged_for_review)
        self.assertEqual(result.feedback_source, "llm")


def make_transport(handler):
    import httpx
    return httpx.MockTransport(handler)


class ExampleDecisionProviderTests(unittest.TestCase):
    def test_jev_env_key_selects_jev_client(self):
        import example_usage

        with patch.dict(os.environ, {"JEV_API_KEY": "jev-test"}, clear=True):
            provider = example_usage.get_decision_provider()

        self.assertIsInstance(provider, JevDecisionClient)
        self.assertEqual(provider.model, "jev-latest")

    def test_main_model_mode_returns_none(self):
        import example_usage

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(example_usage.get_decision_provider())


if __name__ == "__main__":
    unittest.main()
