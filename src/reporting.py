"""Research-report text shared by the Word download and the text summary."""

from code_mapping import render_code_landscape
from next_data_plan import build_next_data_plan
from reflexivity import REFLEXIVITY_CHECKS, REFLEXIVITY_FIELDS
from research_records import render_research_records

CODING_METHODS_NOTE = (
    "编码方法：第一轮用过程编码，名称是过程性短语；直接采用受访者原话或隐喻的，标为实境编码。"
    "接着做同义归并，这时才写定义、包含和排除。再做代码映射（代码全集、类别、更少的类别、概念），然后归纳主题。"
    "设计依据仍是《质性研究入门指南》。这一段只说明这次用了哪些编码操作。"
)


KIND_LABELS = {
    "pattern": "案例内模式",
    "information_breakpoint": "信息断点",
    "counterexample": "反例",
    "evidence_gap": "证据缺口",
    "candidate_mechanism": "候选机制",
}
STATEMENT_LABELS = {
    "material_fact": "材料或事件支持",
    "participant_report": "受访者报告",
    "participant_interpretation": "受访者判断",
    "researcher_interpretation": "研究者解释",
    "undetermined": "待判定",
}
VERIFICATION_LABELS = {
    "supported_by_material": "有材料支持",
    "reported_only": "仅有受访者陈述",
    "conflicting": "存在冲突",
    "unknown": "尚未核实",
}
MAX_QUOTES = 4


def _code_map(result):
    return {
        code.get("code_id"): code
        for code in result.get("codes") or []
        if isinstance(code.get("code_id"), int)
    }


def _analysis_sentence(theme, quote) -> str:
    meaning = quote.get("definition") or quote.get("description") or theme.get("description") or "这条原话"
    rationale = (theme.get("rationale") or "").strip()
    if rationale:
        return f"分析：{meaning}。这些材料放在一起，是因为{rationale}"
    description = theme.get("description") or ""
    return (
        f"分析：{meaning}。它被归入「{theme.get('name') or '该主题'}」。"
        f"{description}"
    )


def _take_quotes(code_ids, code_by_id, used, limit):
    taken = []
    for code_id in code_ids:
        if len(taken) >= limit:
            break
        code = code_by_id.get(code_id)
        if not code:
            continue
        excerpt = str(code.get("excerpt") or "").strip()
        if not excerpt or excerpt in used:
            continue
        taken.append({
            "code_id": code_id,
            "excerpt": excerpt,
            "description": code.get("name") or code.get("description") or "",
            "definition": code.get("definition") or "",
            "statement_type": code.get("statement_type") or "",
            "verification_status": code.get("verification_status") or "",
            "source": code.get("source") or {},
            "speaker": code.get("speaker") or "",
        })
        used.add(excerpt)
    return taken


def _pack_lengths(packs, fixed_length):
    quote_length = 0
    other_length = fixed_length
    for pack in packs:
        other_length += len(pack["name"]) + len(pack["description"]) + len(pack.get("rationale") or "") + 120
        for quote in pack["supports"] + pack["counters"]:
            quote_length += len(quote["excerpt"])
            other_length += len(_analysis_sentence(pack, quote))
    return quote_length, other_length


def _within_budget(packs, fixed_length) -> bool:
    quote_length, other_length = _pack_lengths(packs, fixed_length)
    return quote_length <= other_length


def assign_theme_quotes(result):
    """Choose thick-description quotes without reusing an excerpt."""
    code_by_id = _code_map(result)
    used = set()
    packs = []
    for theme in result.get("final_themes") or []:
        counter_ids = [
            code_id for code_id in theme.get("counterexample_code_ids") or []
            if isinstance(code_id, int)
        ]
        counter_set = set(counter_ids)
        support_ids = [
            code_id for code_id in theme.get("code_ids") or []
            if isinstance(code_id, int) and code_id not in counter_set
        ]
        pack = {
            "name": theme.get("name") or "",
            "description": theme.get("description") or "",
            "rationale": theme.get("rationale") or "",
            "kind": theme.get("kind") or "",
            "supports": _take_quotes(support_ids, code_by_id, used, 1),
            "counters": _take_quotes(counter_ids, code_by_id, used, 1),
            "support_ids": support_ids,
            "counter_ids": counter_ids,
            "open_questions": list(theme.get("open_questions") or []),
            "uncertain": theme.get("uncertain") or "",
            "code_ids": list(theme.get("code_ids") or []),
            "plain_codes": [str(code) for code in theme.get("codes") or []],
            "category_names": list(theme.get("category_names") or []),
        }
        packs.append(pack)
    fixed_length = 480
    for field in ("supports", "counters"):
        progressed = True
        while progressed:
            progressed = False
            for pack in packs:
                current = pack[field]
                if len(current) >= MAX_QUOTES:
                    continue
                source = pack["support_ids"] if field == "supports" else pack["counter_ids"]
                while len(current) < MAX_QUOTES:
                    extra = _take_quotes(source, code_by_id, used, 1)
                    if not extra:
                        break
                    current.append(extra[0])
                    if _within_budget(packs, fixed_length):
                        progressed = True
                        break
                    current.pop()
    return packs


def _quote_line(quote) -> str:
    speaker = STATEMENT_LABELS.get(quote.get("statement_type"), "待判定")
    verification = VERIFICATION_LABELS.get(quote.get("verification_status"), "")
    label = quote.get("description") or "未命名编码"
    detail = f"{speaker}；{verification}" if verification else speaker
    source = quote.get("source") or {}
    context = ""
    if source.get("source_id"):
        context = (f"；来源 {source['source_id']}；参与者 {source.get('participant_id') or '未知'}；"
                   f"时间 {source.get('recorded_at') or '未知'}；事件 {source.get('event_id') or '未知'}")
    return f"「{quote['excerpt']}」（{detail}；编码 [{quote['code_id']}] {label}{context}）"


def _reflexivity_lines(reflexivity):
    data = reflexivity or {}
    lines = [("h1", "研究者立场")]
    position = str(data.get("position") or "").strip()
    lines.append(("p", position or "（请用两三句话写明你的立场。模型不代写这一段。）"))
    lines.append(("h1", "自反记录"))
    lines.append(("p", "下面这些话来自研究者自己的记录。模型不写、也不读取这一节。"))
    for key, label, *_rest in REFLEXIVITY_FIELDS:
        lines.append(("h2", label))
        lines.append(("p", str(data.get(key) or "").strip() or "（尚未填写）"))
    lines.append(("h2", "定稿前可以自问"))
    for key, label in REFLEXIVITY_CHECKS:
        value = str(data.get(key) or "").strip()
        lines.append(("p", f"{label} {value or '（尚未填写）'}"))
    return lines


def _theme_lines(result, packs, interpretations):
    lines = [("h1", f"最终主题（{len(packs)}）")]
    if not packs:
        lines.append(("p", "本次分析没有生成主题。"))
        return lines
    code_by_id = _code_map(result)
    for index, pack in enumerate(packs, 1):
        lines.append(("h2", f"{index}. {pack['name']}"))
        kind = KIND_LABELS.get(pack["kind"], "待分类")
        lines.append(("p", f"类型：{kind}"))
        if pack.get("category_names"):
            lines.append(("p", f"来自类别：{'、'.join(pack['category_names'])}"))
        if pack["description"]:
            lines.append(("p", pack["description"]))
        if pack["rationale"]:
            lines.append(("p", f"备忘录：{pack['rationale']}"))
        lines.append(("h3", "厚描引文"))
        if pack["supports"]:
            for quote in pack["supports"]:
                lines.append(("p", _quote_line(quote)))
                lines.append(("p", _analysis_sentence(pack, quote)))
        elif pack["plain_codes"] and not pack["code_ids"]:
            for code in pack["plain_codes"]:
                lines.append(("bullet", code))
            lines.append(("p", "这些编码没有单独的逐字原话，所以上面只列出编码短语。"))
        else:
            lines.append(("p", "这次没有尚未在别处用过、又可单独引用的原话。"))
        lines.append(("h3", "诠释"))
        note = ""
        if interpretations and index - 1 < len(interpretations):
            note = str(interpretations[index - 1] or "").strip()
        lines.append(("p", note or "（请在此写下你的诠释。模型不代写这一段。）"))
        lines.append(("h3", "反例与边界"))
        if pack["counters"]:
            for quote in pack["counters"]:
                lines.append(("p", _quote_line(quote)))
                lines.append(("p", f"这条反例来自编码 [{quote['code_id']}]，不并进上面的正文。"))
        elif pack["counter_ids"]:
            lines.append(("p", f"反例编码：{', '.join(map(str, pack['counter_ids']))}。这些编码没有可用的逐字摘录。"))
        else:
            lines.append(("p", "这次没有标出反例。"))
        lines.append(("h3", "未决问题"))
        questions = []
        for question in [*pack["open_questions"], pack["uncertain"]]:
            text = str(question or "").strip()
            if text and text not in questions:
                questions.append(text)
        for code_id in pack["code_ids"]:
            code = code_by_id.get(code_id) or {}
            text = str(code.get("open_question") or "").strip()
            if text and text not in questions:
                questions.append(text)
        if questions:
            lines.extend(("bullet", question) for question in questions)
        else:
            lines.append(("p", "这次没有写下未决问题。"))
    return lines


def _codebook_lines(result):
    lines = [("h1", "编码簿")]
    codes = [
        code for code in result.get("codes") or []
        if code.get("merged_into") is None and (code.get("description") or code.get("name"))
    ]
    if not codes:
        lines.append(("p", "完整定义见本地编码簿文件；这次结果里没有可单列的主编码。"))
        return lines
    lines.append(("p", "每条主编码保留定义、包含、排除和原文位置。正例若已在主题中引用，这里不再重复。"))
    for code in codes:
        label = code.get("name") or code.get("description")
        lines.append(("h2", f"[{code.get('code_id')}] {label}"))
        lines.append(("p", f"方法：{code.get('method') or '过程编码'}"))
        lines.append(("p", f"定义：{code.get('definition') or '（尚未写定义）'}"))
        related = [code_id for code_id in code.get("related_code_ids") or [] if isinstance(code_id, int)]
        if related:
            lines.append(("p", f"相关编码：{', '.join(map(str, related))}"))
        if str(code.get("note") or "").strip():
            lines.append(("p", f"备注：{code['note']}"))
        lines.append(("p", f"包含：{code.get('include') or '（尚未写）'}"))
        lines.append(("p", f"排除：{code.get('exclude') or '（尚未写）'}"))
        if code.get("source_start") is not None:
            lines.append(("p", f"原文位置：字符 {code['source_start']}–{code.get('source_end')}"))
        members = code.get("merged_from") or []
        if members:
            lines.append(("p", f"并入来源：{', '.join(map(str, members))}"))
    return lines


def _plan_lines(result):
    plan = result.get("next_data_plan")
    if not plan:
        plan = build_next_data_plan(
            result.get("codes"), result.get("final_themes"), result.get("memos"),
        )
    groups = plan.get("groups") or {}
    lines = [("h1", "下一轮可以收集什么")]
    if not any(groups.values()):
        lines.append(("p", "这次没有留下待核查问题。"))
        return lines
    for name in ("需补访角色", "需查材料", "需核对信息"):
        entries = groups.get(name) or []
        if not entries:
            continue
        lines.append(("h2", name))
        lines.extend(("bullet", entry) for entry in entries)
    for retired in result.get("retired_findings") or []:
        theme = retired["theme"]
        lines.append(("h2", f"未纳入最终主题但保留的线索：{theme['name']}"))
        lines.append(("p", f"移出理由：{retired['reason']}；反例编码：{theme.get('counterexample_code_ids') or []}"))
    return lines


def _focus_lines():
    return [
        ("h1", "聚焦"),
        ("p", "这一节留给研究者。模型不写、也不读取。"),
        ("p", "这次最想留下的主题（以下三处是写作提示，可按研究实际增减）："),
        ("p", "1. （请填写）"),
        ("p", "2. （请填写）"),
        ("p", "3. （请填写）"),
        ("p", "一句论断：（请写一句可能还需要修改的话。）"),
        ("p", "支持这句论断的编码：（请填写编号）"),
        ("p", "不支持这句论断的编码：（请填写编号）"),
    ]


def _landscape_lines(result):
    code_map = result.get("code_map") or (result.get("generation") or {}).get("code_map")
    text = render_code_landscape(result.get("codes"), code_map)
    lines = []
    for raw in text.splitlines():
        if raw.startswith("# "):
            lines.append(("h1", raw[2:].strip()))
        elif raw.startswith("- "):
            lines.append(("bullet", raw[2:].strip()))
        elif raw.startswith("  - "):
            lines.append(("bullet", raw[4:].strip()))
        elif raw.strip():
            lines.append(("p", raw.strip()))
    return lines


def _closing_lines(result):
    lines = [("h1", "边界与局限")]
    lines.append(("p", "（请写明这次没有做什么、材料允许你说什么。模型不代写这一节。）"))
    gaps = [
        theme.get("name") for theme in result.get("final_themes") or []
        if theme.get("kind") == "evidence_gap" and theme.get("name")
    ]
    if gaps:
        lines.append(("p", "本次分析已标出的证据缺口："))
        lines.extend(("bullet", name) for name in gaps)
    lines.append(("h1", "所以……"))
    lines.append(("p", "（请写推论，并留下仍值得追问的问题。模型不代写这一节。）"))
    return lines


def report_lines(result, reflexivity=None, interpretations=None):
    """Return ordered report blocks as (kind, text)."""
    status = "本轮主题评估通过" if result.get("accepted") else "本轮自动处理结束"
    score = (result.get("metadata") or {}).get("final_average_score")
    lines = [
        ("title", "访谈质性分析报告"),
        ("p", f"分析编号：{result.get('session_name', '')}"),
    ]
    if result.get("timestamp"):
        lines.append(("p", f"分析时间：{result['timestamp']}"))
    if result.get("case_id"):
        lines.append(("p", f"匿名案例编号：{result['case_id']}"))
    model = (result.get("configuration") or {}).get("model")
    if model:
        lines.append(("p", f"模型：{model}"))
    lines.append(("p", f"状态：{status}"))
    if isinstance(score, (int, float)):
        lines.append(("p", f"平均分：{score:.2f} / 5"))
    if result.get("refinement_iterations") is not None:
        lines.append(("p", f"评估轮次：{result['refinement_iterations']}"))
    lines.append(("p", "分数衡量主题是否像一个有证据的模式。候选机制和诠释仍要由研究者自己写、自己判断。"))
    lines.append(("p", "本轮处理结束不代表理论饱和；类属充分性须附研究者的比较和判断依据。"))
    lines.append(("p", CODING_METHODS_NOTE))
    storyline = result.get("analytic_storyline") if "analytic_storyline" in result else (result.get("generation") or {}).get("analytic_storyline")
    if storyline:
        lines.append(("p", f"本轮主题故事线（草稿，模型所写）：{storyline}"))
    if result.get("storyline_needs_review"):
        lines.append(("p", "主题已经修订，尚无与最终主题对应的故事线；请研究者重新核对整体论证。"))
    note = (result.get("generation") or {}).get("codebook_note")
    if note:
        lines.append(("p", note))
    lines.extend(_reflexivity_lines(reflexivity))
    evaluation = result.get("final_evaluation") or {}
    if evaluation.get("global_feedback"):
        lines.append(("h1", "最终评估"))
        lines.append(("p", evaluation["global_feedback"]))
    flagged = evaluation.get("flagged_themes") or []
    if flagged:
        lines.append(("p", f"建议人工复核（{len(flagged)}）：{'、'.join(flagged)}"))
    packs = assign_theme_quotes(result)
    lines.extend(_theme_lines(result, packs, interpretations))
    lines.extend(_focus_lines())
    lines.extend(_landscape_lines(result))
    lines.extend(_plan_lines(result))
    lines.extend(_codebook_lines(result))
    if result.get("research_workspace"):
        for raw in render_research_records(result["research_workspace"]).splitlines():
            if raw.startswith("# "):
                lines.append(("h1", raw[2:]))
            elif raw.startswith("## "):
                lines.append(("h2", raw[3:]))
            elif raw.startswith("- "):
                lines.append(("bullet", raw[2:]))
            elif raw.strip():
                lines.append(("p", raw))
    lines.extend(_closing_lines(result))
    return lines


def render_summary_text(result, reflexivity=None, interpretations=None) -> str:
    parts = []
    for kind, text in report_lines(result, reflexivity, interpretations):
        if kind == "title":
            parts.append(text)
            parts.append("=" * 40)
        elif kind == "h1":
            parts.append("")
            parts.append(text)
            parts.append("-" * 20)
        elif kind == "h2":
            parts.append("")
            parts.append(text)
        elif kind == "h3":
            parts.append(text)
        elif kind == "bullet":
            parts.append(f"- {text}")
        else:
            parts.append(text)
    return "\n".join(parts).rstrip() + "\n"
