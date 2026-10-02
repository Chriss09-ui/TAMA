"""Deterministic comparison cells and original paragraph/question context."""

import re

from chunking import INTERVIEWER_LABEL_PATTERN, SPEAKER_PATTERN
from evidence import is_matched, matched_codes


DIMENSIONS = {"participant_id": "参与者", "case_id": "案例", "speaker": "说话人标签",
              "event_id": "事件", "recorded_at": "资料时间", "source_id": "材料"}


def evidence_context(result, code):
    source_id = (code.get("source") or {}).get("source_id")
    document = next((item for item in result.get("source_documents") or [] if item["source_id"] == source_id), None)
    start, end = code.get("source_start"), code.get("source_end")
    if (not is_matched(code) or not document or end > len(document["text"])
            or document["text"][start:end] != code["excerpt"]):
        return {"kind": "已保存附近语境（原稿未关联或位置未通过核验）", "text": code.get("context") or "",
                "source_id": source_id, "start": None, "end": None, "complete": False}
    text = document["text"]
    questions = [match.start() for match in SPEAKER_PATTERN.finditer(text)
                 if INTERVIEWER_LABEL_PATTERN.fullmatch(match.group(1).strip())]
    before = [position for position in questions if position <= start]
    if before:
        left = before[-1]
        right = next((position for position in questions if position >= end), len(text))
        kind = "所属问答"
    else:
        breaks = list(re.finditer(r"\n[ \t]*\n", text))
        left = max((match.end() for match in breaks if match.end() <= start), default=0)
        right = next((match.start() for match in breaks if match.start() >= end), len(text))
        kind = "所属段落"
    return {"kind": kind, "text": text[left:right], "source_id": source_id,
            "start": left, "end": right, "complete": True}


def build_comparison_matrix(result, dimension="participant_id"):
    if dimension not in DIMENSIONS:
        raise ValueError("不支持的比较维度")
    codes = matched_codes(result.get("codes") or [])
    by_id = {code["code_id"]: code for code in codes}
    groups = {}
    def group(source, speaker=None):
        value = speaker if dimension == "speaker" else source.get(dimension)
        source_id = source.get("source_id") or "未标记"
        key = ("known", value) if value else ("unknown", source_id)
        label = value or f"未知（材料 {source_id}）"
        return groups.setdefault(key, (label, []))[1]

    # Empty source rows remain visible; missing evidence never implies opposition.
    for source in (result.get("research_workspace") or {}).get("sources") or []:
        if dimension != "speaker":
            group(source)
    for code in codes:
        source = code.get("source") or {}
        group(source, code.get("speaker")).append(code["code_id"])
    themes = result.get("final_themes") or []
    rows = []
    for label, group_ids in groups.values():
        ids = set(group_ids)
        cells = []
        for theme in themes:
            counter_ids = set(theme.get("counterexample_code_ids") or [])
            if theme.get("kind") == "counterexample":
                counter_ids.update(theme.get("code_ids") or [])
            support = sorted(ids & (set(theme.get("code_ids") or []) - counter_ids))
            counter = sorted(ids & counter_ids)
            state = "混合" if support and counter else "支持" if support else "反对" if counter else "未涉及"
            cells.append({"theme_id": theme.get("theme_id") or theme["name"], "theme": theme["name"],
                          "state": state, "supporting_code_ids": support, "contradicting_code_ids": counter})
        rows.append({"label": label, "code_ids": sorted(ids), "cells": cells})
    sources = {(code.get("source") or {}).get("source_id") for code in codes}
    sources.update(item["source_id"] for item in result.get("source_documents") or [])
    sources.update(item.get("source_id") for item in (result.get("research_workspace") or {}).get("sources") or [])
    participants = {(code.get("source") or {}).get("participant_id") for code in codes}
    cases = {(code.get("source") or {}).get("case_id") for code in codes}
    evidence = {((code.get("source") or {}).get("source_id"), code["source_start"], code["source_end"]) for code in codes}
    counts = {"codes": len(by_id), "evidence": len(evidence), "sources": len(sources - {None, ""}),
              "participants": len(participants - {None, ""}), "cases": len(cases - {None, ""}),
              "unknown_participants": sum(not (code.get("source") or {}).get("participant_id") for code in codes),
              "unknown_cases": sum(not (code.get("source") or {}).get("case_id") for code in codes)}
    return {"dimension": dimension, "rows": rows, "counts": counts}
