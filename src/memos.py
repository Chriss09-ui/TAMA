"""Analytic memos for why a set of codes became one theme."""

HUMAN_OPEN = "**[human]**"
HUMAN_CLOSE = "**[/human]**"
HUMAN_PLACEHOLDER = "在此写下你的判断。标记之间的文字不会进入模型。"
def _ids(values) -> list[int]:
    found = []
    for value in values or []:
        if isinstance(value, int) and value not in found:
            found.append(value)
    return found


def attach_operation_rationales(themes, operations):
    """Use a refinement operation's rationale when the new theme has none."""
    by_name = {theme.get("name"): theme for theme in themes}
    for operation in operations or []:
        new_theme = operation.get("new_theme") or {}
        theme = by_name.get(new_theme.get("name"))
        rationale = str(operation.get("rationale") or "").strip()
        if theme is not None and rationale and not str(theme.get("rationale") or "").strip():
            theme["rationale"] = rationale
    return themes


def build_theme_memos(themes, previous=None, iteration=0):
    """Build one memo per theme and keep counterexamples on the theme."""
    prior = {item.get("theme"): item for item in previous or []}
    memos = []
    for theme in themes:
        name = theme.get("name") or ""
        earlier = prior.get(name) or {}
        supporting = _ids(theme.get("code_ids"))
        contradicting = _ids([
            *(theme.get("counterexample_code_ids") or []),
            *(earlier.get("contradicting_code_ids") or []),
        ])
        theme["counterexample_code_ids"] = contradicting
        rationale = str(theme.get("rationale") or earlier.get("rationale") or "").strip()
        uncertain = str(theme.get("uncertain") or earlier.get("uncertain") or "").strip()
        theme["rationale"] = rationale
        theme["uncertain"] = uncertain
        human_note = str(earlier.get("human_note") or "").strip()
        if human_note == HUMAN_PLACEHOLDER:
            human_note = ""
        memos.append({
            "theme": name,
            "supporting_code_ids": supporting,
            "rationale": rationale,
            "contradicting_code_ids": contradicting,
            "uncertain": uncertain,
            "human_note": human_note,
            "iteration": iteration,
        })
    return memos


def memos_for_model(memos):
    """Drop researcher-written notes before any model prompt."""
    visible = []
    for memo in memos or []:
        visible.append({
            "theme": memo.get("theme") or "",
            "supporting_code_ids": _ids(memo.get("supporting_code_ids")),
            "rationale": memo.get("rationale") or "",
            "contradicting_code_ids": _ids(memo.get("contradicting_code_ids")),
            "uncertain": memo.get("uncertain") or "",
        })
    return visible


def render_memos_markdown(memos) -> str:
    lines = [
        "# 分析备忘录",
        "",
        "标记 **[human]** 与 **[/human]** 之间的文字只留给研究者，不会进入模型。",
        "",
    ]
    for memo in memos or []:
        lines.append(f"## {memo.get('theme') or '未命名主题'}")
        lines.append("")
        supporting = ", ".join(map(str, memo.get("supporting_code_ids") or [])) or "无"
        contradicting = ", ".join(map(str, memo.get("contradicting_code_ids") or [])) or "无"
        lines.append(f"- 支持编码：{supporting}")
        lines.append(f"- 为何构成同一模式：{memo.get('rationale') or '（模型没有写下理由）'}")
        lines.append(f"- 反证编码：{contradicting}")
        lines.append(f"- 不确定：{memo.get('uncertain') or '无'}")
        lines.append("")
        lines.append(HUMAN_OPEN)
        lines.append(memo.get("human_note") or HUMAN_PLACEHOLDER)
        lines.append(HUMAN_CLOSE)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def human_notes_from_markdown(markdown: str) -> dict:
    notes = {}
    current = None
    capture = False
    bucket: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            capture = False
            bucket = []
            continue
        if current and line.strip() == HUMAN_OPEN:
            capture = True
            bucket = []
            continue
        if current and line.strip() == HUMAN_CLOSE:
            text = "\n".join(bucket).strip()
            notes[current] = "" if text == HUMAN_PLACEHOLDER else text
            capture = False
            continue
        if capture:
            bucket.append(line)
    return notes


def apply_human_notes(memos, notes):
    updated = []
    for memo in memos or []:
        item = dict(memo)
        if item.get("theme") in notes:
            item["human_note"] = notes[item["theme"]]
        updated.append(item)
    return updated
