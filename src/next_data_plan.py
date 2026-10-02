"""Collect checks and interpretation limits for the submitted material."""

MATERIAL_MARKERS = ("来源", "摘录", "说话人", "编号", "原文", "匹配", "位置")
CONFLICT_MARKERS = ("矛盾", "冲突", "反例", "反证", "不同说法")
GROUP_ORDER = ("原文与来源核对", "矛盾与反例检查", "解释边界")
COLLECTION_ACTIONS = ("补访", "访谈提纲", "招募", "去问", "安排采访", "联系受访者", "重新访谈", "理论抽样", "询问负责人", "寻找受访者", "采访提纲")


def material_review_text(text: str) -> str:
    """Do not present legacy collection instructions as current analysis tasks."""
    cleaned = str(text or "").strip()
    if any(action in cleaned for action in COLLECTION_ACTIONS):
        return "当前材料尚无法回答此问题，请核对相关原文与解释边界。"
    return cleaned


def classify_question(text: str) -> str:
    if any(marker in text for marker in CONFLICT_MARKERS):
        return "矛盾与反例检查"
    if any(marker in text for marker in MATERIAL_MARKERS):
        return "原文与来源核对"
    return "解释边界"


def build_next_data_plan(codes, themes, memos, workspace=None, corpus_review=None) -> dict:
    """Group unresolved questions. Nothing already recorded is dropped."""
    items = []
    seen = set()

    def add(text, source, code_ids=None):
        cleaned = material_review_text(text)
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        items.append({
            "text": cleaned,
            "source": source,
            "group": classify_question(cleaned),
            "code_ids": list(code_ids or []),
        })

    for code in codes or []:
        add(code.get("open_question"), f"编码 {code.get('code_id')}", [code.get("code_id")])
    for theme in themes or []:
        name = theme.get("name") or "未命名主题"
        for question in theme.get("open_questions") or []:
            add(question, f"主题 {name}")
        counters = [code_id for code_id in theme.get("counterexample_code_ids") or [] if isinstance(code_id, int)]
        if counters:
            add(
                f"主题「{name}」有反例编码 {', '.join(map(str, counters))}，需要核对这些说法是否构成边界。",
                f"主题 {name}",
            )
    for memo in memos or []:
        add(memo.get("uncertain"), f"备忘录 {memo.get('theme')}")
    for comparison in (workspace or {}).get("comparisons") or []:
        add(comparison.get("open_question"), f"比较 {comparison.get('code_ids')}")
    for finding in (corpus_review or {}).get("findings") or []:
        add(finding.get("reason"), f"全文回查 · {finding.get('kind')}", [finding["code_id"]])

    groups = {name: [] for name in GROUP_ORDER}
    for item in items:
        groups[item["group"]].append(item["text"])
    tasks = (workspace or {}).get("sampling_tasks") or []
    return {"groups": groups, "items": items, "tasks": tasks,
            "markdown": render_next_data_plan(groups, tasks)}


def render_next_data_plan(groups, tasks=None) -> str:
    lines = [
        "# 当前材料复核",
        "",
        "这里汇总本次文稿中的原文核对、矛盾和解释边界。材料没有交代的内容保持未知；不生成采访或收集任务。",
        "",
    ]
    for name in GROUP_ORDER:
        lines.append(f"## {name}")
        lines.append("")
        entries = groups.get(name) or []
        if not entries:
            lines.append("（这次没有归入这一组的问题）")
        else:
            lines.extend(f"- {entry}" for entry in entries)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
