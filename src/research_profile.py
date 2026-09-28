"""Study-specific focus kept separate from reusable analysis instructions."""

from dataclasses import dataclass
from typing import Sequence


ENTERPRISE_FOCUS = (
    "capability", "boundary", "key_event", "internal_evidence",
    "evidence_holder", "public_status", "public_location",
    "information_breakpoint", "role_difference", "counterexample", "unknown",
)
ENTERPRISE_QUESTION = "企业具体能力如何形成内部证据、由谁掌握、如何成为公开信息，链条中有哪些断点？"


@dataclass(frozen=True)
class ResearchProfile:
    name: str = "generic"
    research_question: str = ""
    focus_areas: tuple[str, ...] = ()

    def prompt_section(self) -> str:
        if not self.research_question and not self.focus_areas:
            return ""
        lines = ["研究聚焦（仅作为分析方向；不得强行套用到无关材料）："]
        if self.research_question:
            lines.append(f"研究问题：{self.research_question}")
        if self.focus_areas:
            lines.append(f"关注点：{'、'.join(self.focus_areas)}")
        return "\n".join(lines) + "\n\n"


def resolve_profile(
    name: str = "generic",
    research_question: str = "",
    focus_areas: Sequence[str] | None = None,
) -> ResearchProfile:
    if name not in ("generic", "enterprise_evidence"):
        raise ValueError(f"Unknown research profile: {name}")
    if name == "enterprise_evidence":
        question = research_question.strip() or ENTERPRISE_QUESTION
        areas = tuple(focus_areas) if focus_areas is not None else ENTERPRISE_FOCUS
    else:
        question = research_question.strip()
        areas = tuple(focus_areas or ())
    return ResearchProfile(name, question, tuple(dict.fromkeys(area.strip() for area in areas if area.strip())))
