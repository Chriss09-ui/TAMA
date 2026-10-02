"""Shared original-text checks, independent of real-world verification."""

from hashlib import sha256


EVIDENCE_LABELS = {
    "matched": "已匹配", "missing": "缺少摘录",
    "mismatch": "匹配失败", "unverified": "尚未核验",
}
REVIEW_KIND_LABELS = {
    "missed_support": "遗漏支持", "counterexample": "反例", "context_limit": "解释条件",
}


def value(record, key, default=None):
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def set_value(record, key, item):
    if isinstance(record, dict):
        record[key] = item
    else:
        setattr(record, key, item)


def source_document(source_id: str, text: str) -> dict:
    normalized = text.strip()
    return {"source_id": source_id, "text": normalized,
            "sha256": sha256(normalized.encode("utf-8")).hexdigest()}


def validate_documents(documents) -> list[dict]:
    parsed, seen = [], set()
    for item in documents or []:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("原文记录格式无效")
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise ValueError("原文来源编号缺失或重复")
        expected = source_document(source_id, item["text"])
        if item.get("sha256") != expected["sha256"] or item["text"] != expected["text"]:
            raise ValueError("原文内容与校验值不一致")
        parsed.append(expected)
        seen.add(source_id)
    return parsed


def is_matched(code) -> bool:
    excerpt = value(code, "excerpt")
    start, end = value(code, "source_start"), value(code, "source_end")
    return (value(code, "evidence_status") == "matched"
            and isinstance(excerpt, str) and bool(excerpt)
            and type(start) is int and type(end) is int
            and start >= 0 and end == start + len(excerpt))


def matched_codes(codes):
    return [code for code in codes or [] if is_matched(code)]


def validate_evidence(codes, documents):
    """Recompute statuses from bodies and offsets, never trust imported flags."""
    texts = {item["source_id"]: item["text"] for item in validate_documents(documents)}
    for code in codes:
        excerpt = value(code, "excerpt")
        source = value(code, "source", {})
        text = texts.get(value(source, "source_id"))
        start, end = value(code, "source_start"), value(code, "source_end")
        if not isinstance(excerpt, str) or not excerpt:
            status = "mismatch" if value(code, "evidence_status") == "mismatch" else "missing"
        elif text is None:
            status = "unverified"
        elif (type(start) is int and type(end) is int and 0 <= start < end <= len(text)
              and end == start + len(excerpt) and text[start:end] == excerpt):
            status = "matched"
        else:
            status = "mismatch"
        set_value(code, "evidence_status", status)
        if status != "matched":
            reason = f"原文证据{EVIDENCE_LABELS[status]}，不作为主题支持；需人工核对。"
            question = str(value(code, "open_question", "") or "")
            if reason not in question:
                set_value(code, "open_question", "；".join(filter(None, (question, reason))))
    allowed = {value(code, "code_id") for code in matched_codes(codes)}
    by_id = {value(code, "code_id"): code for code in codes}
    for code in codes:
        parent = value(code, "merged_into")
        if parent is not None and (parent not in by_id or
                ((value(code, "code_id") in allowed) != (parent in allowed))):
            set_value(code, "merged_into", None)
    for code in codes:
        set_value(code, "merged_from", [item for item in value(code, "merged_from", [])
                                        if item in by_id and value(by_id[item], "merged_into") == value(code, "code_id")])
    return codes


def link_themes_to_codes(themes, codes):
    """Sanitize explicit references, label fallbacks and every merged member."""
    by_id = {value(code, "code_id"): code for code in matched_codes(codes)}
    by_label = {}
    for code_id, code in by_id.items():
        for label in (value(code, "description"), value(code, "name")):
            if label:
                by_label.setdefault(label, []).append(code_id)

    def expand(ids):
        result = []
        for code_id in ids:
            if type(code_id) is not int or code_id not in by_id:
                continue
            code = by_id[code_id]
            root = value(code, "merged_into")
            root = root if root in by_id else code_id
            result.extend([root, code_id, *value(by_id[root], "merged_from", [])])
        return list(dict.fromkeys(item for item in result if type(item) is int and item in by_id))

    linked = []
    for original in themes or []:
        theme = dict(original)
        raw_ids = theme.get("code_ids") or []
        ids = expand(raw_ids)
        # A rejected explicit reference must not be rescued by a shared label.
        if not raw_ids:
            ids = expand([item for label in theme.get("codes") or [] for item in by_label.get(label, [])])
        # A counterexample belongs to its original excerpt, not every merged sibling.
        counters = list(dict.fromkeys(item for item in theme.get("counterexample_code_ids") or []
                                     if type(item) is int and item in by_id))
        theme.update(code_ids=ids, counterexample_code_ids=counters,
                     codes=[value(by_id[item], "description") for item in ids],
                     open_questions=list(theme.get("open_questions") or []),
                     kind=theme.get("kind") or "pattern")
        if not any(item not in counters for item in ids):
            theme["kind"] = "evidence_gap"
            question = "该发现缺少已匹配原文的支持编码，需人工核对。"
            if question not in theme["open_questions"]:
                theme["open_questions"].append(question)
        linked.append(theme)
    return linked
