"""Typed decision primitives shared by all decision providers.

A decision request evaluates one `state` against a set of typed questions and
returns one normalized answer per question key. Agents author the questions;
providers own the transport.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, model_validator

QuestionKind = Literal["noul", "choice", "score"]

_KIND_PAYLOAD_FIELD = {
    "noul": "probability",
    "choice": "choice",
    "score": "score",
}


class DecisionQuestion(BaseModel):
    """One typed question about a state, authored by the calling agent."""

    key: str
    kind: QuestionKind
    instructions: str
    # choice: option name -> description (at most 255 options per the Jev API)
    options: Optional[Dict[str, str]] = None
    # score: ordered level descriptions from lowest to highest (2-10 levels)
    scale: Optional[List[str]] = None


class DecisionAnswer(BaseModel):
    """Normalized answer for one question key.

    `score` is the probability-weighted level index, zero-based (level 0 is
    the lowest scale entry); agents convert it to their own rating ranges.
    """

    key: str
    kind: QuestionKind
    probability: Optional[float] = None  # noul: P(yes) in 0-1
    choice: Optional[str] = None  # choice: highest-probability option name
    score: Optional[float] = None  # score: weighted zero-based level index
    confidence: float = 0.0  # 0-1
    probabilities: Optional[Dict[str, float]] = None
    legend: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> "DecisionAnswer":
        field = _KIND_PAYLOAD_FIELD[self.kind]
        if getattr(self, field) is None:
            raise ValueError(f"{self.kind} answer requires '{field}'")
        return self
