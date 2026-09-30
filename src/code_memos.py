"""Analytic memos tied to codes, written before themes exist."""

from memos import HUMAN_CLOSE, HUMAN_OPEN, HUMAN_PLACEHOLDER


def _ids(values) -> list[int]:
    found = []
    for value in values or []:
        if isinstance(value, int) and value not in found:
            found.append(value)
    return found


def memos_from_groups(groups, source, on_date):
    memos = []
    for group in groups or []:
        if not isinstance(group, dict):
            continue
        text = str(group.get("memo") or "").strip()
        if not text:
            continue
        memos.append({
            "date": on_date,
            "source": source,
            "subtitle": str(group.get("name") or "").strip(),
            "code_ids": _ids(group.get("code_ids")),
            "text": text,
            "human_note": "",
        })
    return memos


def code_memo_heading(memo, index) -> str:
    ids = ", ".join(map(str, _ids(memo.get("code_ids")))) or "无编号"
    subtitle = str(memo.get("subtitle") or "").strip()
    source = str(memo.get("source") or "备忘").strip()
    title = f"{source} · {ids}"
    if subtitle:
        title = f"{title} · {subtitle}"
    return f"{index}. {title}"


def code_memos_for_model(memos):
    """Drop researcher-written notes before a model prompt."""
    visible = []
    for memo in memos or []:
        text = str(memo.get("text") or "").strip()
        if not text:
            continue
        visible.append({
            "date": memo.get("date") or "",
            "source": memo.get("source") or "",
            "code_ids": _ids(memo.get("code_ids")),
            "text": text,
        })
    return visible


def render_code_memos_markdown(memos) -> str:
    lines = [
        "# 编码备忘录",
        "",
        "这些备忘录写在主题之前，挂在编码上。",
        "标记 **[human]** 与 **[/human]** 之间的文字只留给研究者，不会进入模型。",
        "",
    ]
    if not memos:
        lines.append("这次没有在归并或映射时写下编码备忘录。")
        return "\n".join(lines).rstrip() + "\n"
    for index, memo in enumerate(memos, 1):
        lines.append(f"## {code_memo_heading(memo, index)}")
        lines.append("")
        lines.append(f"- 日期：{memo.get('date') or '未标日期'}")
        ids = ", ".join(map(str, _ids(memo.get("code_ids")))) or "无"
        lines.append(f"- 编码：{ids}")
        lines.append("")
        lines.append(str(memo.get("text") or "").strip() or "（没有写下文字）")
        lines.append("")
        lines.append(HUMAN_OPEN)
        lines.append(memo.get("human_note") or HUMAN_PLACEHOLDER)
        lines.append(HUMAN_CLOSE)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
