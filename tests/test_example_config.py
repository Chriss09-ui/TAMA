"""Example-script provider defaults, overrides, priority and decision mode."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decisions.jev_client import JevDecisionClient
from example_usage import get_model_config


class ExampleModelConfigTests(unittest.TestCase):
    def test_example_selects_deepseek_defaults(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek-test"}, clear=True):
            self.assertEqual(
                get_model_config(),
                ("deepseek-test", "deepseek-flash", "https://api.deepseek.com")
            )

    def test_example_accepts_model_and_endpoint_overrides(self):
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "deepseek-test",
            "DEEPSEEK_MODEL": "deepseek-v4-pro",
            "DEEPSEEK_BASE_URL": "https://custom.example/v1",
        }, clear=True):
            self.assertEqual(
                get_model_config(),
                ("deepseek-test", "deepseek-v4-pro", "https://custom.example/v1")
            )

    def test_example_provider_priority(self):
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "deepseek-test",
            "OPENAI_API_KEY": "openai-test",
        }, clear=True):
            self.assertEqual(get_model_config()[0], "deepseek-test")

        with patch.dict(os.environ, {
            "MIMO_API_KEY": "mimo-test",
            "DEEPSEEK_API_KEY": "deepseek-test",
        }, clear=True):
            self.assertEqual(get_model_config()[0], "mimo-test")

    def test_example_selects_mimo_with_default_endpoint(self):
        with patch.dict(os.environ, {"MIMO_API_KEY": "mimo-test"}, clear=True):
            self.assertEqual(
                get_model_config(),
                ("mimo-test", "mimo-v2.5-pro", "https://api.xiaomimimo.com/v1")
            )

    def test_example_uses_token_plan_endpoint(self):
        with patch.dict(
            os.environ,
            {"MIMO_API_KEY": "mimo-test", "MIMO_BASE_URL": "https://plan.example/v1"},
            clear=True
        ):
            self.assertEqual(get_model_config()[2], "https://plan.example/v1")

    def test_example_keeps_openai_default(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test"}, clear=True):
            self.assertEqual(get_model_config(), ("openai-test", "gpt-4o", None))

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
