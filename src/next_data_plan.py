"""Collect open questions into a plan for the next round of material."""

ROLE_MARKERS = ("谁", "角色", "访谈", "受访者", "导师", "经理", "同事", "补访", "询问", "负责人")
MATERIAL_MARKERS = ("材料", "文件", "文档", "官网", "合同", "记录", "邮件", "系统", "截图", "原文", "档案")
GROUP_ORDER = ("需补访角色", "需查材料", "需核对信息")


def classify_question(text: str) -> str:
    if any(marker in text for marker in MATERIAL_MARKERS):
        return "需查材料"
    if any(marker in text for marker in ROLE_MARKERS):
        return "需补访角色"
    return "需核对信息"


def build_next_data_plan(codes, themes, memos, workspace=None) -> dict:
    """Group unresolved questions. Nothing already recorded is dropped."""
    items = []
    seen = set()

    def add(text, source):
        cleaned = str(text or "").strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        items.append({
            "text": cleaned,
            "source": source,
            "group": classify_question(cleaned),
        })

    for code in codes or []:
        add(code.get("open_question"), f"编码 {code.get('code_id')}")
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

    groups = {name: [] for name in GROUP_ORDER}
    for item in items:
        groups[item["group"]].append(item["text"])
    tasks = (workspace or {}).get("sampling_tasks") or []
    return {"groups": groups, "items": items, "tasks": tasks,
            "markdown": render_next_data_plan(groups, tasks)}


def render_next_data_plan(groups, tasks=None) -> str:
    lines = [
        "# 下一轮可以收集什么",
        "",
        "这里汇总这次分析留下的待核查问题、备忘录里的不确定和反例。它是下一轮抽样和访谈提纲的原料，不判断材料是否已经足够。",
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
    for task in tasks or []:
        lines.extend([f"## {task['kind']}：{task['target']}", "",
                      f"- 类属：{task.get('category_id') or '未关联'}",
                      f"- 缺口：{task['gap']}", f"- 不同解释：{task.get('alternatives') or '待明确'}",
                      f"- 选择理由：{task['rationale']}", f"- 开放追问：{task.get('questions') or '待填写'}",
                      f"- 预期改变：{task.get('expected_change') or '一般核查，不推定理论目的'}",
                      f"- 状态：{task['status']}；分析后果：{task.get('outcome') or '尚未回填'}", ""])
    return "\n".join(lines).rstrip() + "\n"
