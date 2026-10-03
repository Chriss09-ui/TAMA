"""Project .env persistence using disposable files and synthetic keys."""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api_settings import get_api_setting, read_api_key, write_api_key


class ApiSettingsTests(unittest.TestCase):
    def setUp(self):
        env_dir = tempfile.TemporaryDirectory()
        self.addCleanup(env_dir.cleanup)
        self.env_path = Path(env_dir.name) / ".env"
        env_patch = patch("api_settings.ENV_PATH", self.env_path)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_missing_file_uses_environment_and_defaults(self):
        with patch.dict(os.environ, {"MIMO_API_KEY": "env-test"}, clear=True):
            self.assertEqual(read_api_key("MiMo"), "env-test")
            self.assertEqual(get_api_setting("CUSTOM_MODEL", "fallback"), "fallback")
        self.assertFalse(self.env_path.exists())

    def test_file_settings_override_environment_without_expansion(self):
        self.env_path.write_text('MIMO_API_KEY="file-${EXTERNAL}-test"\n', encoding="utf-8")
        with patch.dict(os.environ, {"MIMO_API_KEY": "env-test", "EXTERNAL": "expanded"}, clear=True):
            self.assertEqual(read_api_key("MiMo"), "file-${EXTERNAL}-test")

    def test_overwrite_preserves_other_settings_and_comments(self):
        self.env_path.write_text(
            '# existing comment\nMIMO_API_KEY="old-test"\nCUSTOM_MODEL="vendor-model"\nDEEPSEEK_API_KEY="other-test"\n',
            encoding="utf-8",
        )
        write_api_key("MiMo", " new-test ")
        self.assertEqual(read_api_key("MiMo"), "new-test")
        self.assertEqual(read_api_key("DeepSeek"), "other-test")
        self.assertEqual(get_api_setting("CUSTOM_MODEL"), "vendor-model")
        self.assertIn("# existing comment", self.env_path.read_text(encoding="utf-8"))

    def test_clearing_key_stays_empty_after_reload_with_environment_present(self):
        write_api_key("MiMo", "old-test")
        write_api_key("MiMo", "")
        with patch.dict(os.environ, {"MIMO_API_KEY": "env-test"}, clear=True):
            self.assertEqual(read_api_key("MiMo"), "")
        self.assertNotIn("old-test", self.env_path.read_text(encoding="utf-8"))

    def test_all_provider_keys_round_trip_without_affecting_each_other(self):
        values = {"MiMo": "mimo-test", "DeepSeek": "deepseek-test", "自定义": "custom-test", "Jev": "jev-test"}
        for provider, value in values.items():
            write_api_key(provider, value)
        for provider, value in values.items():
            self.assertEqual(read_api_key(provider), value)

    def test_quoted_special_characters_round_trip(self):
        value = "test-'quote'-#hash-$dollar-${literal}-\\backslash"
        write_api_key("MiMo", value)
        self.assertEqual(read_api_key("MiMo"), value)

    @unittest.skipIf(os.name == "nt", "POSIX file permissions")
    def test_saved_file_is_private_to_current_user(self):
        write_api_key("MiMo", "test-key")
        self.assertEqual(stat.S_IMODE(self.env_path.stat().st_mode), 0o600)

    def test_failed_write_leaves_previous_key_unchanged(self):
        write_api_key("MiMo", "old-test")
        with patch("api_settings.set_key", side_effect=OSError("synthetic failure")):
            with self.assertRaises(OSError):
                write_api_key("MiMo", "new-test")
        self.assertEqual(read_api_key("MiMo"), "old-test")


if __name__ == "__main__":
    unittest.main()
