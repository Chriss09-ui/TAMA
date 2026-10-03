"""Discover local reports without treating their contents as trusted evidence."""

import json
from datetime import datetime
from pathlib import Path

from analysis_records import restore_report


def report_label(result: dict) -> str:
    title = str(result.get("case_id") or "访谈分析")
    try:
        date = datetime.fromisoformat(str(result.get("timestamp", "")))
    except ValueError:
        return str(result.get("case_id") or result["session_name"])
    return f"{title} · {date:%m/%d %H:%M}"


def local_reports(directory: Path) -> list[dict]:
    reports = []
    for path in directory.glob("*/00_final_results.json"):
        if path.is_symlink() or path.parent.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if (not isinstance(payload, dict) or not isinstance(payload.get("session_name"), str)
                    or not payload["session_name"] or not isinstance(payload.get("final_themes"), list)):
                continue
            reports.append({"session_name": payload["session_name"], "label": report_label(payload),
                            "timestamp": str(payload.get("timestamp", "")), "path": path})
        except (OSError, ValueError, TypeError):
            continue
    return sorted(reports, key=lambda item: (item["timestamp"], item["session_name"]), reverse=True)


def load_library_report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and "codes" not in payload and isinstance(payload.get("generation"), dict):
        if not isinstance(payload.get("session_name"), str) or not payload["session_name"]:
            raise ValueError("报告缺少名称")
        themes = payload.get("final_themes")
        if not isinstance(themes, list):
            raise ValueError("报告缺少主题")
        for theme in themes:
            if (not isinstance(theme, dict) or not isinstance(theme.get("name"), str)
                    or not isinstance(theme.get("description"), str)
                    or not isinstance(theme.get("codes", []), list)
                    or not all(isinstance(code, str) for code in theme.get("codes", []))):
                raise ValueError("旧版报告主题格式无效")
        # Older exports have no verifiable source records. Keep this reading
        # view separate from the evidence and researcher editing tools.
        return {"session_name": payload["session_name"], "timestamp": payload.get("timestamp", ""),
                "final_themes": themes, "legacy_read_only": True}
    return restore_report(payload)
