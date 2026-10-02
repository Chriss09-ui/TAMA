"""HTTP client for the TypeSafe Jev System One decision API.

Wire format reference: https://docs.typesafe.ai/api
This module is the single adaptation point if the API evolves: question
serialization lives in `_serialize_question`, answer parsing in `_parse_answers`.
"""

import math
import random
import time
from typing import Any, Dict, Optional

import httpx

from decisions.base import (
    DecisionAnswer,
    DecisionProvider,
    DecisionProviderError,
    DecisionQuestion,
    State,
)

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"
DEFAULT_MODEL = "jev-latest"
RETRY_STATUS_CODES = {429, 529}


class JevDecisionClient(DecisionProvider):
    """Posts one state with all questions to the Jev System One endpoint."""

    name = "jev"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
            transport=transport,
        )
        self.model = model or DEFAULT_MODEL
        self.max_retries = max_retries
        self.last_usage: Optional[Dict[str, int]] = None

    def ask(self, state: State, questions: Dict[str, DecisionQuestion]) -> Dict[str, DecisionAnswer]:
        payload = {
            "state": state,
            "model": self.model,
            "questions": {key: self._serialize_question(q) for key, q in questions.items()},
        }
        body = self._post(payload)
        return self._parse_answers(body, questions)

    def _serialize_question(self, question: DecisionQuestion) -> Dict[str, Any]:
        if question.kind == "noul":
            return {"type": "noul", "instructions": question.instructions}
        if question.kind == "choice":
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": question.options,
            }
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": question.scale,
        }

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        attempt = 0
        while True:
            try:
                response = self._client.post(f"{self.base_url}/systemone", json=payload)
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise DecisionProviderError(f"Jev 决策接口连接失败：{exc}") from exc
                time.sleep(self._backoff(attempt))
                attempt += 1
                continue
            if response.status_code in RETRY_STATUS_CODES:
                if attempt >= self.max_retries:
                    raise DecisionProviderError(
                        f"Jev 决策接口繁忙（HTTP {response.status_code}），重试 {self.max_retries} 次后仍失败。"
                    )
                time.sleep(self._backoff(attempt))
                attempt += 1
                continue
            if response.status_code == 401:
                raise DecisionProviderError(
                    "Jev 决策接口认证失败（HTTP 401）：请检查 JEV_API_KEY 是否有效。", permanent=True,
                )
            if response.status_code == 422:
                raise DecisionProviderError(
                    "Jev 决策接口拒绝了请求（HTTP 422）：请检查问题构造。", permanent=True,
                )
            if response.status_code >= 400:
                raise DecisionProviderError(f"Jev 决策接口返回 HTTP {response.status_code}。")
            body = response.json()
            self.last_usage = body.get("usage")
            return body

    def _parse_answers(self, body: Any, questions: Dict[str, DecisionQuestion]) -> Dict[str, DecisionAnswer]:
        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise DecisionProviderError("Jev 决策接口没有返回预期的 answers 结构。")
        raw_answers = body["answers"]
        answers: Dict[str, DecisionAnswer] = {}
        for key, question in questions.items():
            raw = raw_answers.get(key)
            if not isinstance(raw, dict):
                raise DecisionProviderError(f"Jev 决策接口缺少问题「{key}」的答案。")
            if question.kind == "noul":
                probability = _clamp(_required_float(raw, "noul", key), 0.0, 1.0)
                answers[key] = DecisionAnswer(
                    key=key, kind="noul", probability=probability,
                    # The API defines no separate confidence for noul; distance
                    # from coin-flip is the derived certainty of the decision.
                    confidence=max(probability, 1.0 - probability),
                )
            elif question.kind == "choice":
                choice = raw.get("choice")
                if question.options is not None and choice not in question.options:
                    raise DecisionProviderError(f"Jev 决策接口对问题「{key}」返回了未知选项：{choice}")
                answers[key] = DecisionAnswer(
                    key=key, kind="choice", choice=choice,
                    confidence=_clamp(_required_float(raw, "confidence", key), 0.0, 1.0),
                    probabilities=_float_map(raw.get("probabilities")),
                )
            else:
                answers[key] = DecisionAnswer(
                    key=key, kind="score", score=_required_float(raw, "score", key),
                    confidence=_clamp(_required_float(raw, "confidence", key), 0.0, 1.0),
                    probabilities=_float_map(raw.get("probabilities")),
                    legend=_str_map(raw.get("legend")),
                )
        return answers

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(0.5 * (2 ** attempt), 8.0) + random.uniform(0, 0.25)


def _required_float(raw: Dict[str, Any], field: str, key: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionProviderError(f"Jev 决策接口对问题「{key}」返回了无法解析的 {field}。")
    try:
        number = float(value)
    except OverflowError as exc:
        raise DecisionProviderError(f"Jev 决策接口对问题「{key}」返回了无法解析的 {field}。") from exc
    if not math.isfinite(number):
        raise DecisionProviderError(f"Jev 决策接口对问题「{key}」返回了非有限的 {field}。")
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _float_map(raw: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw, dict):
        return None
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _str_map(raw: Any) -> Optional[Dict[str, str]]:
    if not isinstance(raw, dict):
        return None
    return {str(k): str(v) for k, v in raw.items()}
