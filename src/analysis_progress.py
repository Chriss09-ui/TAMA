"""Safe progress events shared by the analysis workers and their UI."""

from dataclasses import dataclass


DEFAULT_CONCURRENCY = 8
MAX_CONCURRENCY = 16


@dataclass(frozen=True)
class ProgressEvent:
    kind: str
    stage: str
    batch_id: str = ""
    total: int = 0
    unit: str = "任务"
    index: int = 0
    failed: bool = False
    iteration: int = 0
    max_iterations: int = 0


@dataclass(frozen=True)
class ProgressRecord:
    elapsed_seconds: float
    message: str


@dataclass(frozen=True)
class PhaseRecord:
    phase: str
    elapsed_seconds: float
    completed: int
    total: int
    failed: int
    unit: str
    iteration: int = 0
    state: str = "complete"


_PHASES = {
    "准备运行", "切分材料", "提取编码", "归并编码", "映射编码", "归纳主题",
    "合并候选主题", "全文回查", "校验全文回查证据", "评估主题", "修订主题",
    "整理本次结果",
}
_PHASE_ALIASES = {"完整评估": "评估主题", "生成评估反馈": "评估主题"}
_UNITS = {"任务", "片段", "主题", "批次", "组", "项"}


def normalize_phase(stage: str) -> str:
    """Keep material-derived names out of progress summaries and history."""
    phase = stage.split(" · ", 1)[0].strip()
    phase = _PHASE_ALIASES.get(phase, phase)
    return phase if phase in _PHASES else "处理中"


def normalize_unit(unit: str) -> str:
    return unit if unit in _UNITS else "任务"
