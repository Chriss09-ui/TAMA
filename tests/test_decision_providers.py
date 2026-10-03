import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decisions.base import DecisionProviderError, DecisionQuestion
from decisions.jev_client import JevDecisionClient
from decisions.llm_client import LLMDecisionClient


SCALE = ["1 分：较差", "2 分：较弱", "3 分：一般", "4 分：良好", "5 分：优秀"]

QUESTIONS = {
    "coverage": DecisionQuestion(
        key="coverage", kind="score", scale=SCALE, instructions="评估覆盖度。",
    ),
    "needs_check": DecisionQuestion(
        key="needs_check", kind="noul", instructions="该主题是否需要修订？",
    ),
    "route": DecisionQuestion(
        key="route", kind="choice", options={"add": "增加", "split": "拆分"},
        instructions="选择修订操作。",
    ),
}


def fake_llm_response(data):
    message = SimpleNamespace(content=json.dumps(data))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class LLMDecisionClientTests(unittest.TestCase):
    def test_non_finite_decision_numbers_are_rejected(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            for key, field in (("coverage", "score"), ("coverage", "confidence"), ("needs_check", "probability")):
                with self.subTest(value=value, key=key, field=field), patch("decisions.llm_client.OpenAI") as client_cls:
                    data = {"answers": {"coverage": {"score": 3, "confidence": 0.8},
                                        "needs_check": {"probability": 0.1, "confidence": 0.8},
                                        "route": {"choice": "add", "confidence": 0.8}}}
                    data["answers"][key][field] = value
                    client_cls.return_value.chat.completions.create.return_value = fake_llm_response(data)
                    with self.assertRaises(DecisionProviderError):
                        LLMDecisionClient(api_key="test").ask({}, QUESTIONS)

    def test_ask_makes_one_json_mode_call_per_state(self):
        with patch("decisions.llm_client.OpenAI") as client_cls:
            client = client_cls.return_value
            client.chat.completions.create.return_value = fake_llm_response({"answers": {
                "coverage": {"score": 3, "confidence": 0.8},
                "needs_check": {"probability": 0.7, "confidence": 0.6},
                "route": {"choice": "split", "confidence": 0.9},
            }})
            answers = LLMDecisionClient(api_key="test", model="mimo-v2.6-pro").ask(
                {"theme": "日常安排"}, QUESTIONS,
            )

        create = client.chat.completions.create
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(create.call_args.kwargs["response_format"], {"type": "json_object"})
        prompt = create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("日常安排", prompt)
        for expected in ("coverage", "needs_check", "route", "评估覆盖度", "可选答案"):
            self.assertIn(expected, prompt)
        self.assertEqual(answers["coverage"].score, 3)
        self.assertEqual(answers["coverage"].confidence, 0.8)
        self.assertEqual(answers["needs_check"].probability, 0.7)
        self.assertEqual(answers["route"].choice, "split")

    def test_out_of_range_values_are_clamped(self):
        with patch("decisions.llm_client.OpenAI") as client_cls:
            client_cls.return_value.chat.completions.create.return_value = fake_llm_response({"answers": {
                "coverage": {"score": 99, "confidence": 1.5},
                "needs_check": {"probability": -0.2, "confidence": 0.5},
                "route": {"choice": "add", "confidence": 0.5},
            }})
            answers = LLMDecisionClient(api_key="test").ask({"theme": "x"}, QUESTIONS)

        self.assertEqual(answers["coverage"].score, 4)  # clamped to top of 5-level scale
        self.assertEqual(answers["coverage"].confidence, 1.0)
        self.assertEqual(answers["needs_check"].probability, 0.0)

    def test_missing_confidence_defaults_to_zero(self):
        with patch("decisions.llm_client.OpenAI") as client_cls:
            client_cls.return_value.chat.completions.create.return_value = fake_llm_response({"answers": {
                "coverage": {"score": 3},
                "needs_check": {"probability": 0.5, "confidence": 0.5},
                "route": {"choice": "add", "confidence": 0.5},
            }})
            answers = LLMDecisionClient(api_key="test").ask({"theme": "x"}, QUESTIONS)

        self.assertEqual(answers["coverage"].confidence, 0.0)

    def test_missing_answer_is_a_recoverable_error(self):
        with patch("decisions.llm_client.OpenAI") as client_cls:
            client_cls.return_value.chat.completions.create.return_value = fake_llm_response(
                {"answers": {}},
            )
            with self.assertRaises(DecisionProviderError) as ctx:
                LLMDecisionClient(api_key="test").ask({"theme": "x"}, QUESTIONS)

        self.assertFalse(ctx.exception.permanent)

    def test_unknown_choice_is_a_recoverable_error(self):
        with patch("decisions.llm_client.OpenAI") as client_cls:
            client_cls.return_value.chat.completions.create.return_value = fake_llm_response({"answers": {
                "coverage": {"score": 3, "confidence": 0.5},
                "needs_check": {"probability": 0.5, "confidence": 0.5},
                "route": {"choice": "merge", "confidence": 0.5},
            }})
            with self.assertRaises(DecisionProviderError) as ctx:
                LLMDecisionClient(api_key="test").ask({"theme": "x"}, QUESTIONS)

        self.assertFalse(ctx.exception.permanent)


def jev_client(handler, **kwargs):
    return JevDecisionClient(
        api_key="jev-test", transport=httpx.MockTransport(handler), **kwargs,
    )


class JevDecisionClientTests(unittest.TestCase):
    def test_non_finite_decision_numbers_are_rejected(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            for key, field in (("coverage", "score"), ("coverage", "confidence"), ("needs_check", "noul")):
                with self.subTest(value=value, key=key, field=field):
                    client = jev_client(lambda request: httpx.Response(200))
                    data = {"answers": {"coverage": {"score": 3, "confidence": 0.8},
                                        "needs_check": {"noul": 0.1},
                                        "route": {"choice": "add", "confidence": 0.8}}}
                    data["answers"][key][field] = value
                    with self.assertRaises(DecisionProviderError):
                        client._parse_answers(data, QUESTIONS)

    def test_ask_posts_expected_request_and_parses_answers(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={
                "model": "jev-latest",
                "answers": {
                    "coverage": {
                        "score": 3.2, "confidence": 0.8,
                        "legend": {"0": SCALE[0], "4": SCALE[4]},
                        "probabilities": {"2": 0.2, "3": 0.6, "4": 0.2},
                    },
                    "needs_check": {"noul": 0.95},
                    "route": {
                        "choice": "split", "confidence": 0.81,
                        "probabilities": {"add": 0.19, "split": 0.81},
                    },
                },
                "usage": {"input_tokens": 120, "output_tokens": 8},
            })

        answers = jev_client(handler).ask({"theme": "日常安排"}, QUESTIONS)

        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertTrue(request.url.path.endswith("/systemone"))
        self.assertEqual(request.headers["Authorization"], "Bearer jev-test")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "jev-latest")
        self.assertEqual(body["state"], {"theme": "日常安排"})
        self.assertEqual(
            body["questions"]["needs_check"],
            {"type": "noul", "instructions": "该主题是否需要修订？"},
        )
        self.assertEqual(body["questions"]["coverage"]["criteria"], SCALE)
        self.assertEqual(body["questions"]["route"]["criteria"], {"add": "增加", "split": "拆分"})

        self.assertEqual(answers["coverage"].score, 3.2)
        self.assertEqual(answers["coverage"].confidence, 0.8)
        self.assertEqual(answers["coverage"].legend, {"0": SCALE[0], "4": SCALE[4]})
        self.assertEqual(answers["needs_check"].probability, 0.95)
        self.assertEqual(answers["needs_check"].confidence, 0.95)
        self.assertEqual(answers["route"].choice, "split")
        self.assertEqual(answers["route"].confidence, 0.81)

    def test_noul_confidence_is_distance_from_uncertainty(self):
        cases = {0.6: 0.6, 0.35: 0.65, 1.0: 1.0}
        for probability, expected in cases.items():
            with self.subTest(probability=probability):
                def handler(request, probability=probability):
                    return httpx.Response(200, json={
                        "answers": {"needs_check": {"noul": probability}},
                    })

                answers = jev_client(handler).ask(
                    "s", {"needs_check": QUESTIONS["needs_check"]},
                )
                self.assertEqual(answers["needs_check"].confidence, expected)

    def test_429_is_retried_with_backoff_then_succeeds(self):
        attempts = []

        def handler(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx.Response(429)
            return httpx.Response(200, json={"answers": {"needs_check": {"noul": 0.9}}})

        with patch("decisions.jev_client.time.sleep") as slept:
            answers = jev_client(handler).ask("s", {"needs_check": QUESTIONS["needs_check"]})

        self.assertEqual(len(attempts), 2)
        self.assertEqual(answers["needs_check"].probability, 0.9)
        self.assertEqual(slept.call_count, 1)

    def test_529_exhausts_retries_and_raises(self):
        attempts = []

        def handler(request):
            attempts.append(request)
            return httpx.Response(529)

        with patch("decisions.jev_client.time.sleep"):
            with self.assertRaises(DecisionProviderError) as ctx:
                jev_client(handler, max_retries=2).ask(
                    "s", {"needs_check": QUESTIONS["needs_check"]},
                )

        self.assertEqual(len(attempts), 3)  # initial attempt + 2 retries
        self.assertFalse(ctx.exception.permanent)

    def test_401_is_permanent_and_actionable(self):
        def handler(request):
            return httpx.Response(401)

        with self.assertRaises(DecisionProviderError) as ctx:
            jev_client(handler).ask("s", {"needs_check": QUESTIONS["needs_check"]})

        self.assertTrue(ctx.exception.permanent)
        self.assertIn("JEV_API_KEY", str(ctx.exception))

    def test_422_is_permanent_without_echoing_body(self):
        def handler(request):
            return httpx.Response(422, json={"detail": "secret-transcript-content"})

        with self.assertRaises(DecisionProviderError) as ctx:
            jev_client(handler).ask("s", {"needs_check": QUESTIONS["needs_check"]})

        self.assertTrue(ctx.exception.permanent)
        self.assertNotIn("secret-transcript-content", str(ctx.exception))

    def test_missing_answer_is_a_recoverable_error(self):
        def handler(request):
            return httpx.Response(200, json={"answers": {}})

        with self.assertRaises(DecisionProviderError) as ctx:
            jev_client(handler).ask("s", {"needs_check": QUESTIONS["needs_check"]})

        self.assertFalse(ctx.exception.permanent)


if __name__ == "__main__":
    unittest.main()
