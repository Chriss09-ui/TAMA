"""Read model settings and save API keys in the project's .env file."""

import os
from pathlib import Path
from threading import Lock

from dotenv import dotenv_values, set_key


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
API_KEY_VARIABLES = {
    "MiMo": "MIMO_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
    "自定义": "CUSTOM_API_KEY",
    "Jev": "JEV_API_KEY",
}
_ENV_LOCK = Lock()


def get_api_setting(name: str, default: str = "") -> str:
    values = dotenv_values(ENV_PATH, interpolate=False)
    if name in values:
        return values[name] or ""
    return os.getenv(name, default)


def read_api_key(provider: str) -> str:
    return (get_api_setting(API_KEY_VARIABLES[provider], "") or "").strip()


def write_api_key(provider: str, api_key: str) -> None:
    with _ENV_LOCK:
        ENV_PATH.touch(mode=0o600, exist_ok=True)
        # An explicit empty value prevents an old process environment key from returning.
        result = set_key(str(ENV_PATH), API_KEY_VARIABLES[provider], api_key.strip(), quote_mode="always")
        if result[0] is not True:
            raise OSError("Unable to update the project .env file")
        ENV_PATH.chmod(0o600)
