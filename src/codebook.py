"""Codebook merge and the readable codebook export."""

import re


def _label_key(code) -> str | None:
    text = (getattr(code, "name", "") or getattr(code, "description", "") or "").strip()
    if not text:
        return None
    return re.sub(r"\s+", "", text).casefold()


def exact_merge(codes):
    """Collapse codes whose labels match once whitespace is removed."""
    groups = {}
    for code in codes:
        key = _label_key(code)
        if key is None:
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


def _apply_label(code, group) -> None:
    name = str(group.get("name") or "").strip()
    if name:
        code.name = name
        code.description = name
    for field in ("definition", "include", "exclude"):
        value = str(group.get(field) or "").strip()
        if value:
            setattr(code, field, value)
    if _truthy(group.get("in_vivo")):
        code.in_vivo = True


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
        if not all(isinstance(code_id, int) and code_id in active for code_id in ids):
            raise ValueError("编码归并结果无法使用")
        if any(code_id in seen for code_id in ids):
            raise ValueError("编码归并结果无法使用")
        seen.extend(ids)
        parsed.append((ids, group))
    for ids, group in parsed:
        if len(ids) == 1:
            _apply_label(by_id[ids[0]], group)
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
    return codes


def render_codebook(codes, include_excerpts=True) -> str:
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
        lines.append(f"- 版本：{code.get('version') or 1}")
        lines.append(f"- 定义：{code.get('definition') or '（尚未写定义）'}")
        lines.append(f"- 包含：{code.get('include') or '（尚未写）'}")
        lines.append(f"- 排除：{code.get('exclude') or '（尚未写）'}")
        if include_excerpts and code.get("excerpt"):
            lines.append(f"- 正例：{code['excerpt']}")
        if code.get("source_start") is not None:
            lines.append(f"- 原文位置：{code['source_start']}–{code.get('source_end')}")
        members = [code_id for code_id in code.get("merged_from") or [] if code_id in by_id]
        if members:
            lines.append(f"- 并入来源：{', '.join(map(str, members))}")
            for member_id in members:
                member = by_id[member_id]
                detail = member.get("description") or ""
                if include_excerpts and member.get("excerpt"):
                    detail = f"{detail}｜{member['excerpt']}" if detail else member["excerpt"]
                lines.append(f"  - [{member_id}] {detail}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
