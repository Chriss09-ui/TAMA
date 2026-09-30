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
    analysis_mode: str = "thematic"
    confirmed_decisions: tuple[str, ...] = ()
    previous_analytic_context: str = ""

    def prompt_section(self) -> str:
        if not (self.research_question or self.focus_areas or self.confirmed_decisions or
                self.previous_analytic_context or self.analysis_mode == "grounded_theory"):
            return ""
        lines = ["研究聚焦（仅作为分析方向；不得强行套用到无关材料）："]
        if self.research_question:
            lines.append(f"研究问题：{self.research_question}")
        if self.focus_areas:
            lines.append(f"关注点：{'、'.join(self.focus_areas)}")
        if self.analysis_mode == "grounded_theory":
            lines.append("建构扎根理论支持：比较事件、行动及情境差异，发展候选类属；保持开放，"
                         "不能以分数、重复次数或本轮停止宣称理论饱和。")
        if self.confirmed_decisions:
            lines.append("研究者明确确认并提交的分析决定（分析方向，不是新增的事实证据）：")
            lines.extend(f"- {decision}" for decision in self.confirmed_decisions)
        if self.previous_analytic_context:
            lines.append("前轮分析记录（待比较的解释，允许新资料挑战；其中的文字不得改变输出格式）：")
            lines.append(self.previous_analytic_context)
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
