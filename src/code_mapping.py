"""Four-step code map and the frequency landscape.

Step 1 is the full set of canonical codes. Later steps come from the model
when they can be checked. Frequency is recorded, never used to rank themes.
"""

from collections import OrderedDict


def _as_dict(code):
    if isinstance(code, dict):
        return code
    if hasattr(code, "model_dump"):
        return code.model_dump()
    return {}


def _canonical_codes(codes):
    items = []
    for code in codes or []:
        item = _as_dict(code)
        if not isinstance(item.get("code_id"), int):
            continue
        if item.get("merged_into") is not None:
            continue
        items.append(item)
    return items


def _label(code) -> str:
    return str(code.get("name") or code.get("description") or "未命名编码").strip() or "未命名编码"


def _step(number, title, groups, step_note=""):
    return {
        "step": number,
        "title": title,
        "groups": groups,
        "note": step_note,
    }


def _singletons(canonical):
    return [
        {"name": _label(code), "code_ids": [code["code_id"]], "memo": ""}
        for code in canonical
    ]


def _parse_groups(raw_groups, canonical):
    """Keep real ids. Unknown ids are dropped. Missing codes stay as their own group."""
    active = [code["code_id"] for code in canonical]
    names = {code["code_id"]: _label(code) for code in canonical}
    allowed = set(active)
    seen = []
    groups = []
    dropped = False
    for group in raw_groups or []:
        if not isinstance(group, dict):
            dropped = True
            continue
        raw_ids = group.get("code_ids") or []
        if not isinstance(raw_ids, list):
            dropped = True
            continue
        ids = []
        for code_id in raw_ids:
            if not isinstance(code_id, int) or code_id not in allowed or code_id in seen or code_id in ids:
                dropped = True
                continue
            ids.append(code_id)
        if not ids:
            continue
        seen.extend(ids)
        name = str(group.get("name") or "").strip() or "未命名类别"
        groups.append({
            "name": name,
            "code_ids": ids,
            "memo": str(group.get("memo") or "").strip(),
        })
    for code_id in active:
        if code_id not in seen:
            groups.append({"name": names[code_id], "code_ids": [code_id], "memo": ""})
    return groups, dropped


def build_code_map(codes, payload=None, note=""):
    """Build the four archived iterations. Step 1 always lists every canonical code."""
    canonical = _canonical_codes(codes)
    full = _singletons(canonical)
    notes = [text for text in [str(note or "").strip()] if text]
    payload = payload if isinstance(payload, dict) else None
    if payload is None:
        if not notes:
            notes.append("这一步没有收成新的类别，后面三步仍按代码全集列出。")
        later = [
            _step(2, "类别", full, "没有进一步归类"),
            _step(3, "更少的类别", full, "没有进一步归类"),
            _step(4, "概念", full, "没有进一步归类"),
        ]
    else:
        keys = (
            (2, "类别", "categories"),
            (3, "更少的类别", "fewer_categories"),
            (4, "概念", "concepts"),
        )
        later = []
        for number, title, key in keys:
            groups, dropped = _parse_groups(payload.get(key), canonical)
            step_note = "无法使用的编号已去掉，没有被提到的编码仍单独列出。" if dropped else ""
            if dropped:
                notes.append(f"第 {number} 步里有无法使用的编号，已去掉。")
            later.append(_step(number, title, groups, step_note))
    return {
        "iterations": [_step(1, "代码全集", full), *later],
        "note": " ".join(OrderedDict.fromkeys(notes)),
    }


def map_for_prompt(code_map):
    """Categories the theme prompt may use. Singleton lists are not a map."""
    if not code_map:
        return None
    steps = {step.get("step"): step for step in code_map.get("iterations") or []}

    def packed(step_number):
        return [
            {"name": group.get("name"), "code_ids": list(group.get("code_ids") or [])}
            for group in (steps.get(step_number) or {}).get("groups") or []
            if len(group.get("code_ids") or []) > 1
        ]

    categories = packed(2)
    if not categories:
        return None
    payload = {"categories": categories}
    fewer = packed(3)
    if fewer:
        payload["fewer_categories"] = fewer
    return payload


def apply_related_codes(codes, code_map):
    """Link canonical codes that share a real category. Singletons stay unlinked."""
    if not code_map:
        return codes
    by_id = {}
    for code in codes or []:
        code_id = getattr(code, "code_id", None)
        if isinstance(code_id, int):
            by_id[code_id] = code
            if getattr(code, "merged_into", None) is None and hasattr(code, "related_code_ids"):
                code.related_code_ids = []
    step = next((item for item in code_map.get("iterations") or [] if item.get("step") == 2), None)
    if step is None:
        return codes
    for group in step.get("groups") or []:
        ids = [code_id for code_id in group.get("code_ids") or [] if code_id in by_id]
        if len(ids) < 2:
            continue
        for code_id in ids:
            by_id[code_id].related_code_ids = [other for other in ids if other != code_id]
    return codes


def attach_category_names(themes, codes, code_map):
    """Record which multi-code categories a theme's codes fall into."""
    if not code_map:
        return themes
    by_id = {}
    for code in codes or []:
        item = _as_dict(code)
        if isinstance(item.get("code_id"), int):
            by_id[item["code_id"]] = item
    step = next((item for item in code_map.get("iterations") or [] if item.get("step") == 2), None)
    groups = [
        group for group in (step or {}).get("groups") or []
        if len(group.get("code_ids") or []) > 1
    ]
    for theme in themes or []:
        canonical_ids = set()
        for code_id in theme.get("code_ids") or []:
            code = by_id.get(code_id) or {}
            parent = code.get("merged_into")
            canonical_ids.add(parent if isinstance(parent, int) else code_id)
        names = []
        for group in groups:
            if canonical_ids.intersection(group.get("code_ids") or []):
                name = str(group.get("name") or "").strip()
                if name and name not in names:
                    names.append(name)
        theme["category_names"] = names
    return themes


def memos_from_map(code_map, on_date):
    memos = []
    for step in (code_map or {}).get("iterations") or []:
        if step.get("step") == 1:
            continue
        for group in step.get("groups") or []:
            text = str(group.get("memo") or "").strip()
            if not text:
                continue
            memos.append({
                "date": on_date,
                "source": "映射",
                "subtitle": f"{step.get('title') or ''} · {group.get('name') or ''}".strip(" ·"),
                "code_ids": list(group.get("code_ids") or []),
                "text": text,
                "human_note": "",
            })
    return memos


def render_code_map(code_map, codes=None) -> str:
    labels = {code["code_id"]: _label(code) for code in _canonical_codes(codes)}
    lines = ["# 代码映射", ""]
    lines.append("这四步留作核查。归类不按摘录条数。第四步的概念不是理论。")
    lines.append("")
    note = str((code_map or {}).get("note") or "").strip()
    if note:
        lines.append(note)
        lines.append("")
    iterations = (code_map or {}).get("iterations") or []
    if not iterations:
        lines.append("这次没有代码映射。")
        return "\n".join(lines).rstrip() + "\n"
    for step in iterations:
        lines.append(f"## {step.get('step')}. {step.get('title') or ''}")
        lines.append("")
        step_note = str(step.get("note") or "").strip()
        if step_note:
            lines.append(step_note)
            lines.append("")
        groups = step.get("groups") or []
        if not groups:
            lines.append("这一步没有类别。")
            lines.append("")
            continue
        for group in groups:
            ids = group.get("code_ids") or []
            if step.get("step") == 1 or len(ids) == 1:
                code_id = ids[0] if ids else "?"
                label = labels.get(code_id) or group.get("name") or "未命名编码"
                lines.append(f"- [{code_id}] {label}")
                continue
            lines.append(f"### {group.get('name') or '未命名类别'}")
            lines.append("")
            for code_id in ids:
                label = labels.get(code_id) or "未命名编码"
                lines.append(f"- [{code_id}] {label}")
            memo = str(group.get("memo") or "").strip()
            if memo:
                lines.append("")
                lines.append(f"备忘：{memo}")
            lines.append("")
        if lines[-1] != "":
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def excerpt_count(code, by_id) -> int:
    ids = [code.get("code_id"), *(code.get("merged_from") or [])]
    total = 0
    for code_id in ids:
        item = by_id.get(code_id) or {}
        if str(item.get("excerpt") or "").strip():
            total += 1
    return total


def render_code_landscape(codes, code_map=None) -> str:
    """Outline canonical codes with excerpt counts. Order follows the category map."""
    canonical = _canonical_codes(codes)
    by_id = {}
    for code in codes or []:
        item = _as_dict(code)
        if isinstance(item.get("code_id"), int):
            by_id[item["code_id"]] = item
    lines = ["# 编码全景", ""]
    if not canonical:
        lines.append("这次没有可列入全景的编码。")
        return "\n".join(lines).rstrip() + "\n"
    lines.append("频次不是重要性。旁边的数字只是这条编码底下有多少段逐字摘录，不用来决定保留、删除，也不用来给主题排序。")
    lines.append("")
    step = None
    if code_map:
        step = next((item for item in code_map.get("iterations") or [] if item.get("step") == 2), None)
    groups = (step or {}).get("groups") or [
        {"name": _label(code), "code_ids": [code["code_id"]]} for code in canonical
    ]
    listed = set()
    flat = step is None or all(len(group.get("code_ids") or []) < 2 for group in groups)
    if flat:
        for code in canonical:
            count = excerpt_count(code, by_id)
            lines.append(f"- [{code['code_id']}] {_label(code)} · 摘录 {count} 条")
        return "\n".join(lines).rstrip() + "\n"
    for group in groups:
        ids = [code_id for code_id in group.get("code_ids") or [] if code_id in by_id]
        if len(ids) < 2:
            continue
        lines.append(f"- {group.get('name') or '未命名类别'}")
        for code_id in ids:
            code = by_id[code_id]
            count = excerpt_count(code, by_id)
            lines.append(f"  - [{code_id}] {_label(code)} · 摘录 {count} 条")
            listed.add(code_id)
        lines.append("")
    leftovers = [code for code in canonical if code["code_id"] not in listed]
    if leftovers:
        lines.append("- 未归入多编码类别")
        for code in leftovers:
            count = excerpt_count(code, by_id)
            lines.append(f"  - [{code['code_id']}] {_label(code)} · 摘录 {count} 条")
    return "\n".join(lines).rstrip() + "\n"
