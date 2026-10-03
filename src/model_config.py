"""Provider model names used by both CLI and web entry points."""

DEFAULT_MIMO_MODEL = "mimo-v2.6-pro"


def api_model_name(provider: str, displayed_model: str) -> str:
    model = displayed_model.strip()
    if provider == "DeepSeek" and model.casefold() == "deepseek-v4.1-flash":
        return "deepseek-flash"
    return model
