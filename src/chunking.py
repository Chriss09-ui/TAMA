"""Deterministic, structure-aware transcript chunk planning."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from math import ceil, floor
import re
from typing import Dict, List, Literal, Optional, Sequence, Tuple


ChunkStrategy = Literal["fine", "balanced", "economy", "manual"]

AUTO_PROFILES: Dict[str, Tuple[int, int]] = {
    "fine": (1200, 1800),
    "balanced": (1800, 2400),
    "economy": (2600, 3600),
}

ANALYSIS_UNIT_PATTERN = re.compile(r"[\u3400-\u9fff]|[^\s\u3400-\u9fff]+")
SPEAKER_PATTERN = re.compile(r"(?m)^[ \t]*([^\n：:]{1,24})[：:]")
INTERVIEWER_LABEL_PATTERN = re.compile(
    r"^(?:访谈者|采访者|研究者|主持人|提问者|问|q|interviewer|moderator)$",
    re.IGNORECASE,
)
RESPONDENT_LABEL_PATTERN = re.compile(
    r"^(?:受访者|被访者|参与者|回答者|答|a|participant|respondent)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkPlan:
    """Resolved sizing information for one transcript."""

    strategy: ChunkStrategy
    total_characters: int
    estimated_tokens: int
    target_size: int
    hard_limit: int
    planned_chunks: int
    num_chunks: int = 0
    measurement_unit: str = "estimated_tokens"
    boundary_policy: str = "question_paragraph_speaker_sentence"
    manual_chunk_size: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ChunkSpan:
    """A half-open character span selected for one model request."""

    start: int
    end: int
    measured_size: int


@dataclass(frozen=True)
class Boundary:
    offset: int
    priority: int


class LengthMeter:
    """Measure substrings and map sizes back to safe character offsets."""

    def __init__(self, text: str, *, manual: bool):
        self.text = text
        self.ends: List[int] = []
        self.cumulative: List[float] = [0.0]

        if manual:
            for match in ANALYSIS_UNIT_PATTERN.finditer(text):
                self.ends.append(match.end())
                self.cumulative.append(self.cumulative[-1] + 1.0)
        else:
            for offset, character in enumerate(text, 1):
                if character.isspace():
                    continue
                if "\u3400" <= character <= "\u9fff":
                    weight = 0.75
                elif character.isascii():
                    weight = 0.375
                else:
                    weight = 1.25
                self.ends.append(offset)
                self.cumulative.append(self.cumulative[-1] + weight)

    @property
    def total(self) -> int:
        return ceil(self.cumulative[-1]) if self.ends else 0

    def index_at(self, offset: int) -> int:
        return bisect_right(self.ends, offset)

    def measure(self, start: int, end: int) -> int:
        start_index = self.index_at(start)
        end_index = self.index_at(end)
        value = self.cumulative[end_index] - self.cumulative[start_index]
        return ceil(value) if value > 0 else 0

    def offset_for_size(self, start: int, size: float, maximum_end: int) -> int:
        start_index = self.index_at(start)
        target = self.cumulative[start_index] + size
        index = bisect_right(self.cumulative, target, lo=start_index + 1) - 1
        if index <= start_index:
            index = start_index + 1
        return min(self.ends[index - 1], maximum_end)


def estimate_tokens(text: str) -> int:
    """Return a conservative, tokenizer-free length estimate."""
    return LengthMeter(text, manual=False).total


def count_analysis_units(text: str) -> int:
    """Count units using the legacy Chinese-character/other-word rule."""
    return sum(1 for _ in ANALYSIS_UNIT_PATTERN.finditer(text))


def resolve_chunk_strategy(
    strategy: Optional[ChunkStrategy],
    manual_chunk_size: Optional[int],
) -> ChunkStrategy:
    """Resolve legacy explicit sizes and validate strategy/size combinations."""
    resolved: ChunkStrategy = strategy or (
        "manual" if manual_chunk_size is not None else "balanced"
    )
    if resolved == "manual" and manual_chunk_size is None:
        raise ValueError("chunk_size is required in manual mode")
    if resolved != "manual" and manual_chunk_size is not None:
        raise ValueError("chunk_size can only be used in manual mode")
    return resolved


def resolve_chunk_plan(
    text: str,
    strategy: ChunkStrategy = "balanced",
    manual_chunk_size: Optional[int] = None,
) -> ChunkPlan:
    """Resolve a document-specific target size and hard limit."""
    if strategy not in (*AUTO_PROFILES, "manual"):
        raise ValueError(f"Unsupported chunk strategy: {strategy}")

    estimated = estimate_tokens(text)
    if strategy == "manual":
        if manual_chunk_size is None or manual_chunk_size < 1:
            raise ValueError("manual_chunk_size must be positive in manual mode")
        total = count_analysis_units(text)
        target = manual_chunk_size
        hard_limit = manual_chunk_size
        unit = "text_units"
    else:
        target, hard_limit = AUTO_PROFILES[strategy]
        total = estimated
        unit = "estimated_tokens"

    if total == 0:
        planned_chunks = 0
        balanced_target = 0
    else:
        nearest_target_count = floor(total / target + 0.5)
        planned_chunks = max(1, ceil(total / hard_limit), nearest_target_count)
        balanced_target = ceil(total / planned_chunks)

    return ChunkPlan(
        strategy=strategy,
        total_characters=len(text),
        estimated_tokens=estimated,
        target_size=balanced_target,
        hard_limit=hard_limit,
        planned_chunks=planned_chunks,
        measurement_unit=unit,
        manual_chunk_size=manual_chunk_size if strategy == "manual" else None,
    )


def _boundary_candidates(text: str) -> List[Boundary]:
    boundaries: Dict[int, int] = {}

    def add(offset: int, priority: int) -> None:
        if 0 < offset < len(text):
            boundaries[offset] = min(priority, boundaries.get(offset, priority))

    for match in SPEAKER_PATTERN.finditer(text):
        label = re.sub(r"\s+", "", match.group(1))
        if INTERVIEWER_LABEL_PATTERN.match(label):
            add(match.start(), 0)
        elif not RESPONDENT_LABEL_PATTERN.match(label):
            add(match.start(), 2)

    for match in re.finditer(r"\n[ \t]*\n+", text):
        add(match.end(), 1)
    for match in re.finditer(r"[。！？!?；;]+[”’\"']*", text):
        add(match.end(), 3)
    for match in re.finditer(r"\n+", text):
        add(match.end(), 4)

    return [Boundary(offset, priority) for offset, priority in sorted(boundaries.items())]


def _choose_boundary(
    start: int,
    maximum_end: int,
    desired_size: int,
    hard_limit: int,
    meter: LengthMeter,
    candidates: Sequence[Boundary],
) -> int:
    minimum_size = max(1, ceil(desired_size * 0.5))
    preferred_low = max(minimum_size, floor(desired_size * 0.8))
    preferred_high = min(hard_limit, ceil(desired_size * 1.2))
    valid = []

    for boundary in candidates:
        if boundary.offset <= start:
            continue
        if boundary.offset > maximum_end:
            break
        size = meter.measure(start, boundary.offset)
        if minimum_size <= size <= hard_limit:
            in_preferred_window = preferred_low <= size <= preferred_high
            valid.append((boundary, size, in_preferred_window))

    question_boundaries = [item for item in valid if item[0].priority == 0]
    preferred = [item for item in valid if item[2]]
    choices = question_boundaries or preferred or valid
    if choices:
        boundary, _, _ = min(
            choices,
            key=lambda item: (item[0].priority, abs(item[1] - desired_size)),
        )
        return boundary.offset

    return meter.offset_for_size(start, hard_limit, maximum_end)


def _rebalance_small_tail(
    spans: List[ChunkSpan],
    meter: LengthMeter,
    candidates: Sequence[Boundary],
    target_size: int,
    hard_limit: int,
) -> List[ChunkSpan]:
    if len(spans) < 2 or spans[-1].measured_size >= max(1, ceil(target_size * 0.5)):
        return spans

    previous = spans[-2]
    tail = spans[-1]
    combined_size = meter.measure(previous.start, tail.end)
    if combined_size <= hard_limit:
        return spans[:-2] + [ChunkSpan(previous.start, tail.end, combined_size)]

    desired = ceil(combined_size / 2)
    valid = []
    for boundary in candidates:
        if not previous.start < boundary.offset < tail.end:
            continue
        left = meter.measure(previous.start, boundary.offset)
        right = meter.measure(boundary.offset, tail.end)
        if left <= hard_limit and right <= hard_limit:
            valid.append((boundary, left, right))

    if valid:
        boundary, left, right = min(
            valid,
            key=lambda item: (item[0].priority, abs(item[1] - desired)),
        )
        return spans[:-2] + [
            ChunkSpan(previous.start, boundary.offset, left),
            ChunkSpan(boundary.offset, tail.end, right),
        ]

    boundary_offset = meter.offset_for_size(previous.start, desired, tail.end)
    return spans[:-2] + [
        ChunkSpan(previous.start, boundary_offset, meter.measure(previous.start, boundary_offset)),
        ChunkSpan(boundary_offset, tail.end, meter.measure(boundary_offset, tail.end)),
    ]


def plan_chunks(
    text: str,
    strategy: ChunkStrategy = "balanced",
    manual_chunk_size: Optional[int] = None,
) -> Tuple[ChunkPlan, List[ChunkSpan]]:
    """Plan structure-aware, non-overlapping spans for a transcript."""
    normalized = text.strip()
    plan = resolve_chunk_plan(normalized, strategy, manual_chunk_size)
    if not normalized:
        return plan, []

    meter = LengthMeter(normalized, manual=strategy == "manual")
    candidates = _boundary_candidates(normalized)
    spans: List[ChunkSpan] = []
    start = 0

    while meter.measure(start, len(normalized)) > plan.hard_limit:
        maximum_end = meter.offset_for_size(start, plan.hard_limit, len(normalized))
        end = _choose_boundary(
            start,
            maximum_end,
            plan.target_size,
            plan.hard_limit,
            meter,
            candidates,
        )
        if end <= start:
            raise ValueError("Unable to advance while chunking transcript")
        spans.append(ChunkSpan(start, end, meter.measure(start, end)))
        start = end

    spans.append(ChunkSpan(start, len(normalized), meter.measure(start, len(normalized))))
    spans = _rebalance_small_tail(
        spans, meter, candidates, plan.target_size, plan.hard_limit,
    )
    return replace(plan, num_chunks=len(spans)), spans
