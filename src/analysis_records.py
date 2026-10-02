"""Stable, local researcher annotations and reviewable analysis events."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from typing import Literal
from uuid import uuid4

from pydantic import Field, StrictInt, model_validator

from evidence import link_themes_to_codes, matched_codes, validate_evidence
from research_records import ResearchRecord, recorded_now, validate_workspace


SCORABLE_KINDS = ("pattern", "candidate_mechanism")
REVIEW_LABELS = {"pending": "尚未复核", "reviewed": "研究者已复核", "needs_revision": "仍需核查"}


class ThemeNote(ResearchRecord):
    theme_id: str = Field(min_length=1)
    text: str = ""
    researcher: str = ""
    date: str = Field(default_factory=recorded_now)


class AlternativeExplanation(ResearchRecord):
    explanation_id: str = Field(default_factory=lambda: uuid4().hex)
    text: str = Field(min_length=1)
    supporting_code_ids: list[StrictInt] = Field(default_factory=list)
    contradicting_code_ids: list[StrictInt] = Field(default_factory=list)
    boundaries: str = ""
    unresolved: str = ""


class ReviewDecision(ResearchRecord):
    status: Literal["pending", "reviewed", "needs_revision"] = "pending"
    researcher: str = ""
    reason: str = ""
    selected_explanation_id: str = ""
    date: str = ""

    @model_validator(mode="after")
    def explicit_review(self):
        if self.status != "pending" and not (self.researcher and self.reason):
            raise ValueError("复核状态需要研究者与判断理由")
        return self


class ResearcherAnnotations(ResearchRecord):
    theme_notes: list[ThemeNote] = Field(default_factory=list)
    explanations: list[AlternativeExplanation] = Field(default_factory=list)
    review: ReviewDecision = Field(default_factory=ReviewDecision)
    source_versions: dict[str, str] = Field(default_factory=dict)


def scorable_themes(themes):
    return [theme for theme in themes if theme.get("kind", "pattern") in SCORABLE_KINDS
            and theme.get("code_ids")]


def ensure_theme_ids(result):
    """Assign once; identity travels with the theme rather than its list position."""
    result.setdefault("analysis_id", uuid4().hex)
    seen = set()
    for theme in result.get("final_themes") or []:
        if not theme.get("theme_id"):
            identity = {key: theme.get(key) for key in ("name", "kind", "code_ids", "counterexample_code_ids")}
            identity["code_ids"] = sorted(identity["code_ids"] or [])
            identity["counterexample_code_ids"] = sorted(identity["counterexample_code_ids"] or [])
            digest = sha256((result["analysis_id"] + json.dumps(identity, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
            theme["theme_id"] = digest[:24]
        if not isinstance(theme["theme_id"], str) or theme["theme_id"] in seen:
            raise ValueError("主题编号缺失或重复，无法关联研究者诠释")
        seen.add(theme["theme_id"])
    return result


def validate_annotations(result, annotations):
    parsed = ResearcherAnnotations.model_validate(annotations or {})
    themes = {theme["theme_id"] for theme in result.get("final_themes") or []}
    notes = [note.theme_id for note in parsed.theme_notes]
    if len(set(notes)) != len(notes) or not set(notes).issubset(themes):
        raise ValueError("研究者诠释引用了不存在或重复的主题")
    allowed = {code["code_id"] for code in matched_codes(result.get("codes") or [])}
    for explanation in parsed.explanations:
        if not set(explanation.supporting_code_ids + explanation.contradicting_code_ids).issubset(allowed):
            raise ValueError("解释只能关联已匹配原文的证据")
        if set(explanation.supporting_code_ids) & set(explanation.contradicting_code_ids):
            raise ValueError("同一解释的支持与反证需分别记录")
    ids = [item.explanation_id for item in parsed.explanations]
    if len(set(ids)) != len(ids):
        raise ValueError("解释编号重复")
    if parsed.review.selected_explanation_id and parsed.review.selected_explanation_id not in ids:
        raise ValueError("复核选择了不存在的解释")
    return parsed


def interpretations_by_theme(result):
    return {note["theme_id"]: note["text"]
            for note in (result.get("researcher_annotations") or {}).get("theme_notes") or []}


def record_event(events, kind, actor, summary, details=None):
    events.append({"event_id": uuid4().hex, "date": recorded_now(), "kind": kind,
                   "actor": actor, "summary": summary, "details": deepcopy(details or {})})


def update_annotations(result, annotations, include_context=None, confirm_review=False, matrix_dimension=None):
    candidate = deepcopy(result)
    ensure_theme_ids(candidate)
    parsed = validate_annotations(candidate, annotations)
    before = candidate.get("researcher_annotations") or ResearcherAnnotations().model_dump()
    after = parsed.model_dump()
    after["source_versions"] = {item["source_id"]: item["sha256"] for item in candidate.get("source_documents") or []}
    if (not confirm_review and before.get("review", {}).get("status") == "reviewed"
            and (before.get("theme_notes") != after["theme_notes"] or before.get("explanations") != after["explanations"])):
        after["review"]["status"] = "pending"
    candidate["researcher_annotations"] = after
    if confirm_review and after["review"]["status"] == "reviewed":
        candidate.pop("annotations_need_review", None)
    if include_context is not None:
        if type(include_context) is not bool:
            raise ValueError("报告语境选项必须为布尔值")
        candidate.setdefault("report_options", {})["include_context"] = include_context
    if matrix_dimension is not None:
        from evidence_views import DIMENSIONS
        if matrix_dimension not in DIMENSIONS:
            raise ValueError("不支持的比较维度")
        candidate.setdefault("report_options", {})["matrix_dimension"] = matrix_dimension
    if before != after or candidate.get("report_options") != result.get("report_options"):
        record_event(candidate.setdefault("audit_trail", []), "researcher_annotations", "研究者",
                     "保存诠释、解释比较与复核判断",
                     {"before": before, "after": after,
                      "report_options": candidate.get("report_options") or {}})
    return candidate


def restore_report(payload):
    """Restore a report for local review only; never authorize its disk paths."""
    from agents.generation_agent import Code

    if not isinstance(payload, dict) or not isinstance(payload.get("session_name"), str) or not payload["session_name"]:
        raise ValueError("请选择 Threadline 完整结果 JSON")
    candidate = deepcopy(payload)
    for field in ("metadata", "configuration", "report_options"):
        if field in candidate and not isinstance(candidate[field], dict):
            raise ValueError("报告配置或评分结构无效")
    if "accepted" in candidate and type(candidate["accepted"]) is not bool:
        raise ValueError("报告处理状态无效")
    options = candidate.get("report_options") or {}
    from evidence_views import DIMENSIONS
    if (options.get("matrix_dimension", "participant_id") not in DIMENSIONS
            or type(options.get("include_context", False)) is not bool):
        raise ValueError("报告选项无效")
    if not isinstance(candidate.get("codes"), list) or not isinstance(candidate.get("final_themes"), list):
        raise ValueError("报告缺少编码或主题列表")
    candidate["codes"] = [Code.model_validate({"source_chunks": [], **item}).model_dump() for item in candidate["codes"]]
    ids = [code["code_id"] for code in candidate["codes"]]
    if len(set(ids)) != len(ids) or any(code_id < 0 for code_id in ids):
        raise ValueError("编码编号重复或无效")
    for theme in candidate["final_themes"]:
        if not isinstance(theme, dict) or not isinstance(theme.get("name"), str) or not isinstance(theme.get("description"), str):
            raise ValueError("主题格式无效")
    validate_evidence(candidate["codes"], candidate.get("source_documents") or [])
    candidate["final_themes"] = link_themes_to_codes(candidate["final_themes"], candidate["codes"])
    if candidate.get("research_workspace"):
        candidate["research_workspace"] = validate_workspace(candidate["research_workspace"], candidate["codes"]).model_dump()
    ensure_theme_ids(candidate)
    candidate["researcher_annotations"] = validate_annotations(candidate, candidate.get("researcher_annotations")).model_dump()
    versions = candidate["researcher_annotations"]["source_versions"]
    if versions and versions != {item["source_id"]: item["sha256"] for item in candidate.get("source_documents") or []}:
        candidate["researcher_annotations"]["review"]["status"] = "pending"
        candidate["annotations_need_review"] = True
    metadata = candidate.setdefault("metadata", {})
    score = metadata.get("final_average_score")
    metadata["final_average_score"] = score
    if score is None:
        candidate["accepted"] = False
    if score is not None and (type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 5):
        raise ValueError("报告评分格式无效")
    eligible = matched_codes(candidate["codes"])
    metadata.update(valid_code_count=len(eligible), pending_code_count=len(candidate["codes"]) - len(eligible))
    if not eligible or not scorable_themes(candidate["final_themes"]):
        candidate["accepted"] = False
        metadata["final_average_score"] = None
        candidate["stop_reason"] = "no_valid_evidence" if not eligible else "no_valid_themes"
    if any(theme.get("kind") == "evidence_gap" for theme in candidate["final_themes"]):
        candidate["accepted"] = False
    if (candidate.get("corpus_review") or {}).get("status") in ("partial", "failed"):
        candidate["accepted"] = False
        candidate["stop_reason"] = "corpus_review_incomplete"
    candidate.setdefault("accepted", False)
    candidate.setdefault("refinement_iterations", 0)
    candidate["configuration"] = {**(candidate.get("configuration") or {}), "save_final": False, "save_intermediate": False}
    candidate.pop("output_dir", None)
    candidate["restored_report"] = True
    if not isinstance(candidate.get("audit_trail", []), list):
        raise ValueError("审计记录格式无效")
    for event in candidate.get("audit_trail") or []:
        if (not isinstance(event, dict) or not all(isinstance(event.get(key), str) for key in ("event_id", "date", "kind", "actor", "summary"))
                or not isinstance(event.get("details"), dict)):
            raise ValueError("审计事件格式无效")
    return candidate


def annotation_report_lines(result):
    annotations = result.get("researcher_annotations") or {}
    review = annotations.get("review") or {}
    lines = [("h1", "研究者解释比较与复核"),
             ("p", f"研究者复核状态：{REVIEW_LABELS.get(review.get('status'), '尚未复核')}")]
    for explanation in annotations.get("explanations") or []:
        selected = "（研究者采用）" if explanation["explanation_id"] == review.get("selected_explanation_id") else ""
        lines.extend([("h2", explanation["text"] + selected),
                      ("p", f"支持编码：{explanation['supporting_code_ids']}；反证编码：{explanation['contradicting_code_ids']}"),
                      ("p", f"解释边界：{explanation['boundaries'] or '尚未填写'}"),
                      ("p", f"无法解释或待核查：{explanation['unresolved'] or '尚未填写'}")])
    if review.get("reason"):
        lines.append(("p", f"研究者：{review.get('researcher') or '未记录'}；判断理由：{review['reason']}；时间：{review.get('date') or '未记录'}"))
    for retired in result.get("retired_findings") or []:
        theme = retired["theme"]
        lines.append(("bullet", f"保留的历史发现：{theme['name']}；反证 {theme.get('counterexample_code_ids') or []}；修订理由：{retired.get('reason') or '未记录'}"))
    events = result.get("audit_trail") or []
    if events:
        lines.append(("h1", "分析与复核过程"))
        for index, event in enumerate(events, 1):
            lines.append(("p", f"{index}. {event['summary']} · {event['actor']} · {event['date']}"))
            details = event.get("details") or {}
            if event.get("kind") == "researcher_annotations":
                before, after = details.get("before") or {}, details.get("after") or {}
                previous = {note["theme_id"]: note["text"] for note in before.get("theme_notes") or []}
                for note in after.get("theme_notes") or []:
                    if previous.get(note["theme_id"], "") != note["text"]:
                        lines.append(("p", f"主题诠释修改：{previous.get(note['theme_id']) or '原先未填写'} → {note['text'] or '已清空'}"))
                decision = after.get("review") or {}
                if decision.get("reason"):
                    lines.append(("p", f"取舍理由：{decision['reason']}；研究者：{decision.get('researcher') or '未记录'}"))
            elif event.get("kind") == "research_records":
                argument = (details.get("after") or {}).get("argument") or {}
                if argument.get("claim"):
                    lines.append(("p", f"整体论断：{argument['claim']}；边界：{argument.get('boundaries') or '未填写'}；反例影响：{argument.get('counterexample_effect') or '未填写'}"))
            elif event.get("kind") == "refinement":
                for operation in (details.get("plan") or {}).get("operations") or []:
                    lines.append(("p", f"主题调整：{operation.get('operation')}；理由：{operation.get('rationale') or '未记录'}"))
    return lines
