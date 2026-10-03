"""Example-script provider defaults, overrides, priority and decision mode."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from decisions.jev_client import JevDecisionClient
from example_usage import get_model_config


class ExampleModelConfigTests(unittest.TestCase):
    def setUp(self):
        env_dir = tempfile.TemporaryDirectory()
        self.addCleanup(env_dir.cleanup)
        self.env_path = Path(env_dir.name) / ".env"
        env_patch = patch("api_settings.ENV_PATH", self.env_path)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_example_reads_project_env_configuration(self):
        self.env_path.write_text(
            'MIMO_API_KEY="file-test"\nMIMO_MODEL="vendor-model"\nMIMO_BASE_URL="https://custom.example/v1"\n',
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"MIMO_API_KEY": "env-test"}, clear=True):
            self.assertEqual(get_model_config(), ("file-test", "vendor-model", "https://custom.example/v1"))

    def test_empty_project_env_key_disables_old_process_key(self):
        self.env_path.write_text('MIMO_API_KEY=""\n', encoding="utf-8")
        with patch.dict(os.environ, {"MIMO_API_KEY": "env-test", "DEEPSEEK_API_KEY": "deepseek-test"}, clear=True):
            self.assertEqual(get_model_config()[0], "deepseek-test")

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
            "CUSTOM_API_KEY": "custom-test",
            "CUSTOM_MODEL": "vendor-model",
            "CUSTOM_BASE_URL": "https://custom.example/v1",
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
                ("mimo-test", "mimo-v2.6-pro", "https://api.xiaomimimo.com/v1")
            )

    def test_example_uses_token_plan_endpoint(self):
        with patch.dict(
            os.environ,
            {"MIMO_API_KEY": "mimo-test", "MIMO_BASE_URL": "https://plan.example/v1"},
            clear=True
        ):
            self.assertEqual(get_model_config()[2], "https://plan.example/v1")

    def test_example_uses_custom_provider_configuration(self):
        with patch.dict(os.environ, {
            "CUSTOM_API_KEY": " custom-test ",
            "CUSTOM_MODEL": " vendor-model ",
            "CUSTOM_BASE_URL": " https://custom.example/v1 ",
        }, clear=True):
            self.assertEqual(
                get_model_config(), ("custom-test", "vendor-model", "https://custom.example/v1")
            )

    def test_custom_provider_requires_model_and_endpoint(self):
        for field in ("CUSTOM_MODEL", "CUSTOM_BASE_URL"):
            env = {
                "CUSTOM_API_KEY": "custom-test", "CUSTOM_MODEL": "vendor-model",
                "CUSTOM_BASE_URL": "https://custom.example/v1",
            }
            env[field] = " "
            with self.subTest(field=field), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, field):
                    get_model_config()

    def test_openai_key_no_longer_selects_a_provider(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test"}, clear=True):
            with self.assertRaisesRegex(ValueError, "CUSTOM_API_KEY"):
                get_model_config()

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
