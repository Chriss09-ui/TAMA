"""Codebook merge and the readable codebook export."""

import re
from evidence import EVIDENCE_LABELS
from datetime import datetime, timezone


def _label_key(code) -> str | None:
    text = (getattr(code, "name", "") or getattr(code, "description", "") or "").strip()
    if not text:
        return None
    return re.sub(r"\s+", "", text).casefold()


def evidence_key(code):
    key = _label_key(code)
    source_id = getattr(getattr(code, "source", None), "source_id", "")
    if key is None or not code.excerpt or not source_id or code.source_start is None:
        return None
    return (key, source_id, code.source_start, code.source_end, code.excerpt,
            getattr(code, "context", ""), code.statement_type)


def exact_merge(codes):
    """Deduplicate the same evidence; identical labels alone do not mean synonyms."""
    groups = {}
    for code in codes:
        key = evidence_key(code)
        if key is None:
            continue
        if code.merged_into is not None or code.separate_from or code.definition_locked:
            continue
        groups.setdefault(key, []).append(code)
    for group in groups.values():
        if len(group) < 2:
            continue
        canonical = min(group, key=lambda item: item.code_id)
        others = [item for item in group if item.code_id != canonical.code_id]
        canonical.merged_from = list(dict.fromkeys([
            *canonical.merged_from, *(item.code_id for item in others),
        ]))
        canonical.version = max(canonical.version, 2)
        for other in others:
            other.merged_into = canonical.code_id
    return codes


def _truthy(value) -> bool:
    return value in (True, "true", "True", "yes", "是")


def revise_definition(code, values, reason, author="模型建议") -> None:
    fields = ("name", "description", "definition", "include", "exclude")
    before = {field: getattr(code, field) for field in fields}
    for field in fields:
        value = str(values.get(field) or "").strip()
        if value:
            setattr(code, field, value)
    after = {field: getattr(code, field) for field in fields}
    if before != after:
        code.definition_history.append({
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "before": before, "after": after, "reason": reason, "author": author,
        })
        code.version += 1
        code.definition_review = "需复核"
        code.review_note = "定义或标签已改变，请回查原摘录与语境。"


def _apply_label(code, group) -> None:
    values = {}
    name = str(group.get("name") or "").strip()
    if name:
        values.update(name=name, description=name)
    for field in ("definition", "include", "exclude"):
        value = str(group.get(field) or "").strip()
        if value and not getattr(code, "definition_locked", False):
            values[field] = value
    revise_definition(code, values, str(group.get("merge_rationale") or "归并后补充或修订编码定义"))
    note = str(group.get("note") or "").strip()
    if note:
        code.note = note
    if _truthy(group.get("in_vivo")) or code.in_vivo:
        code.in_vivo = True
        if hasattr(code, "method"):
            code.method = "实境编码"


def semantic_merge(codes, groups):
    """Apply model-proposed groups. Raise ValueError when the groups are unusable."""
    by_id = {code.code_id: code for code in codes}
    active = {code.code_id for code in codes if code.merged_into is None}
    seen = []
    parsed = []
    for group in groups or []:
        if not isinstance(group, dict):
            raise ValueError("编码归并结果无法使用")
        raw_ids = group.get("code_ids") or []
        if not isinstance(raw_ids, list) or not raw_ids:
            continue
        ids = list(dict.fromkeys(raw_ids))
        if not all(type(code_id) is int and code_id in active for code_id in ids):
            raise ValueError("编码归并结果无法使用")
        if any(code_id in seen for code_id in ids):
            raise ValueError("编码归并结果无法使用")
        if any(set(by_id[code_id].separate_from).intersection(ids) for code_id in ids):
            raise ValueError("归并违反了研究者已确认的保持分开决定")
        locked = [by_id[code_id] for code_id in ids if by_id[code_id].definition_locked]
        if locked and len(ids) > 1:
            raise ValueError("研究者修改的定义需要复核，不能在本轮自动归并")
        seen.extend(ids)
        parsed.append((ids, group))
    for ids, group in parsed:
        reviews = {item.get("code_id"): item for item in group.get("evidence_reviews") or []
                   if isinstance(item, dict) and isinstance(item.get("code_id"), int)}
        rejected = [code_id for code_id in ids if reviews.get(code_id, {}).get("status") == "does_not_fit"]
        for code_id in rejected:
            by_id[code_id].definition_review = "需复核"
            by_id[code_id].review_note = str(reviews[code_id].get("reason") or "该摘录不适用候选定义，已保持独立。")
        ids = [code_id for code_id in ids if code_id not in rejected]
        if not ids:
            continue
        if len(ids) == 1:
            old_version = by_id[ids[0]].version
            _apply_label(by_id[ids[0]], group)
            if by_id[ids[0]].version != old_version:
                for child_id in by_id[ids[0]].merged_from:
                    by_id[child_id].definition_review = "需复核"
                    by_id[child_id].review_note = "规范编码定义已改变，请回查这一条原始摘录。"
            continue
        example = group.get("example_code_id")
        canonical_id = example if example in ids else min(ids)
        canonical = by_id[canonical_id]
        inherited = list(canonical.merged_from)
        for other_id in ids:
            if other_id == canonical_id:
                continue
            other = by_id[other_id]
            for child_id in other.merged_from:
                child = by_id.get(child_id)
                if child is not None:
                    child.merged_into = canonical_id
                inherited.append(child_id)
            other.merged_from = []
            other.merged_into = canonical_id
            inherited.append(other_id)
        canonical.merged_from = list(dict.fromkeys(inherited))
        canonical.version = max(canonical.version, 2)
        _apply_label(canonical, group)
        for member_id in [canonical_id, *canonical.merged_from]:
            member = by_id[member_id]
            member.definition_review = "需复核"
            member.review_note = str(reviews.get(member_id, {}).get("reason") or
                                    "归并定义需与这条原始摘录及语境重新比较。")
    return codes


def render_codebook(codes) -> str:
    """Render one section per canonical code."""
    by_id = {code.get("code_id"): code for code in codes if isinstance(code.get("code_id"), int)}
    canonical = [code for code in codes if code.get("merged_into") is None]
    lines = ["# 编码簿", ""]
    if not canonical:
        lines.append("这次没有编码。")
        return "\n".join(lines) + "\n"
    for code in canonical:
        label = code.get("name") or code.get("description") or "未命名编码"
        lines.append(f"## [{code.get('code_id')}] {label}")
        lines.append("")
        if code.get("in_vivo"):
            lines.append("（内生编码：名称来自材料中的原话或隐喻）")
            lines.append("")
        lines.append(f"- 方法：{code.get('method') or '过程编码'}")
        lines.append(f"- 原文匹配：{EVIDENCE_LABELS.get(code.get('evidence_status'), '尚未核验')}")
        lines.append(f"- 版本：{code.get('version') or 1}")
        lines.append(f"- 定义：{code.get('definition') or '（尚未写定义）'}")
        lines.append(f"- 包含：{code.get('include') or '（尚未写）'}")
        lines.append(f"- 排除：{code.get('exclude') or '（尚未写）'}")
        lines.append(f"- 定义复核：{code.get('definition_review') or '未复核'}；{code.get('review_note') or ''}")
        source = code.get("source") or {}
        if source.get("source_id"):
            lines.append(f"- 来源：{source['source_id']}；参与者：{source.get('participant_id') or '未知'}；"
                         f"时间：{source.get('recorded_at') or '未知'}；事件：{source.get('event_id') or '未知'}")
        if code.get("context"):
            lines.append(f"- 相邻语境：{code['context']}")
        for change in code.get("definition_history") or []:
            lines.append(f"- 定义变更（{change.get('date')}）：{change.get('before')} → {change.get('after')}；"
                         f"理由：{change.get('reason')}；记录者：{change.get('author')}")
        related = [code_id for code_id in code.get("related_code_ids") or [] if isinstance(code_id, int)]
        if related:
            lines.append(f"- 相关编码：{', '.join(map(str, related))}")
        if str(code.get("note") or "").strip():
            lines.append(f"- 备注：{code['note']}")
        if code.get("excerpt"):
            lines.append(f"- 正例：{code['excerpt']}")
        if code.get("source_start") is not None:
            lines.append(f"- 原文位置：{code['source_start']}–{code.get('source_end')}")
        members = [code_id for code_id in code.get("merged_from") or [] if code_id in by_id]
        if members:
            lines.append(f"- 并入来源：{', '.join(map(str, members))}")
            for member_id in members:
                member = by_id[member_id]
                detail = member.get("description") or ""
                if member.get("excerpt"):
                    detail = f"{detail}｜{member['excerpt']}" if detail else member["excerpt"]
                lines.append(f"  - [{member_id}] {detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
