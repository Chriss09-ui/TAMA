import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from example_usage import get_model_config


class DeepSeekConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
