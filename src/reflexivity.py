"""Local reflexivity notes. These records are never added to model prompts."""

from pathlib import Path
import re


FORMAT_MARKER = "<!-- threadline-reflexivity:v2 -->"


REFLEXIVITY_FIELDS = (
    ("surprise", "什么让我惊喜？", "用来暴露你原先的假设。"),
    ("interest", "什么让我感兴趣？", "用来暴露你的立场。"),
    ("trouble", "什么困扰我？", "用来暴露你的偏见。"),
)
REFLEXIVITY_CHECKS = (
    ("noticed", "我注意到了什么？"),
    ("why_noticed", "为什么注意到它？"),
    ("interpretation", "我如何诠释？"),
    ("how_know", "我怎么知道这个诠释站得住？"),
)
POSITION_FIELD = ("position", "研究者立场", "写两三句即可。报告首页会引用这一节。")


def _sections():
    yield POSITION_FIELD[0], POSITION_FIELD[1]
    for key, label, *_rest in REFLEXIVITY_FIELDS:
        yield key, label
    for key, label in REFLEXIVITY_CHECKS:
        yield key, label


def empty_reflexivity() -> dict:
    return {key: "" for key, _label in _sections()}


def render_reflexivity(data: dict) -> str:
    lines = [
        "# 研究者自反",
        "",
        FORMAT_MARKER,
        "",
        "这些记录只保存在本机，不会发送给模型。",
        "",
    ]
    for key, label in _sections():
        text = (data.get(key) or "").strip()
        longest = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
        fence = "`" * max(3, longest + 1)
        lines.append(f"## {label}")
        lines.append("")
        lines.extend([f"{fence}text", text, fence])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_reflexivity(markdown: str) -> dict:
    data = empty_reflexivity()
    labels = {label: key for key, label in _sections()}
    current = None
    bucket: list[str] = []
    modern = FORMAT_MARKER in markdown.splitlines()[:4]
    fence = None

    def flush() -> None:
        if current in data:
            data[current] = "\n".join(bucket).strip()

    for line in markdown.splitlines():
        if fence is not None:
            if line == fence:
                fence = None
            else:
                bucket.append(line)
            continue
        opening = re.fullmatch(r"(`{3,})text", line) if modern else None
        if current is not None and not any(bucket) and opening:
            fence = opening.group(1)
            continue
        if line.startswith("## ") and line[3:].strip() in labels:
            flush()
            current = labels.get(line[3:].strip())
            bucket = []
            continue
        if current is not None:
            bucket.append(line)
    flush()
    return data


def load_reflexivity(path: Path) -> dict:
    if not path.is_file():
        return empty_reflexivity()
    return parse_reflexivity(path.read_text(encoding="utf-8"))


def save_reflexivity(path: Path, data: dict) -> None:
    path.write_text(render_reflexivity(data), encoding="utf-8")
