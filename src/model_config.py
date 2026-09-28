"""Provider model names used by both CLI and web entry points."""


def api_model_name(provider: str, displayed_model: str) -> str:
    model = displayed_model.strip()
    if provider == "DeepSeek" and model.casefold() == "deepseek-v4.1-flash":
        return "deepseek-flash"
    return model
