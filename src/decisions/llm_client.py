"""OpenAI-compatible JSON-mode implementation of the decision interface.

Works with every provider the agents already support (MiMo, DeepSeek, OpenAI).
Confidence values are self-reported by the model, not calibrated; they are a
weak signal kept in the audit trail until a calibrated provider is used.
"""

import json
from typing import Any, Dict, Optional

from openai import OpenAI

from decisions.base import DecisionAnswer, DecisionProvider, DecisionProviderError, DecisionQuestion, State
from prompts import LLM_DECISION_SYSTEM_PROMPT, build_llm_decision_prompt


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class LLMDecisionClient(DecisionProvider):
    """Answers typed decision questions in one JSON-mode chat call per state."""

    name = "llm-json"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def ask(self, state: State, questions: Dict[str, DecisionQuestion]) -> Dict[str, DecisionAnswer]:
        prompt = self._build_prompt(state, questions)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LLM_DECISION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
        except DecisionProviderError:
            raise
        except Exception as exc:
            raise DecisionProviderError(f"决策模型调用失败：{type(exc).__name__}: {exc}") from exc
        return self._parse_answers(data, questions)

    def _build_prompt(self, state: State, questions: Dict[str, DecisionQuestion]) -> str:
        return build_llm_decision_prompt(state, questions)

    def _parse_answers(self, data: Any, questions: Dict[str, DecisionQuestion]) -> Dict[str, DecisionAnswer]:
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise DecisionProviderError("决策模型没有返回预期的 answers 结构。")
        raw_answers = data["answers"]
        answers: Dict[str, DecisionAnswer] = {}
        for key, question in questions.items():
            raw = raw_answers.get(key)
            if not isinstance(raw, dict):
                raise DecisionProviderError(f"决策模型缺少问题「{key}」的答案。")
            confidence = _as_float(raw.get("confidence"), key, default=0.0)
            confidence = _clamp(confidence, 0.0, 1.0)
            if question.kind == "noul":
                probability = _clamp(_as_float(raw.get("probability"), key, default=0.0), 0.0, 1.0)
                answers[key] = DecisionAnswer(
                    key=key, kind="noul", probability=probability, confidence=confidence,
                )
            elif question.kind == "choice":
                choice = raw.get("choice")
                if question.options is not None and choice not in question.options:
                    raise DecisionProviderError(f"决策模型对问题「{key}」返回了未知选项：{choice}")
                answers[key] = DecisionAnswer(
                    key=key, kind="choice", choice=choice, confidence=confidence,
                )
            else:
                scale = question.scale or []
                upper = float(len(scale) - 1) if scale else None
                score = _as_float(raw.get("score"), key, default=0.0)
                if upper is not None:
                    score = _clamp(score, 0.0, upper)
                answers[key] = DecisionAnswer(
                    key=key, kind="score", score=score, confidence=confidence,
                )
        return answers
def _as_float(value: Any, key: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionProviderError(f"决策模型对问题「{key}」返回了无法解析的数值。")
    return float(value)
