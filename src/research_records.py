"""Source context and researcher-owned records for a continuing study."""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, Field, StrictBool, StrictInt, model_validator


def recorded_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResearchRecord(BaseModel):
    model_config = {"str_strip_whitespace": True}


class SourceMetadata(ResearchRecord):
    source_id: str = ""
    case_id: str = ""
    participant_id: str = ""
    data_type: Literal["访谈", "观察笔记", "已有文本", "其他文字材料"] = "访谈"
    recorded_at: str = ""
    setting: str = ""
    event_id: str = ""


class AnalyticDecision(ResearchRecord):
    decision_id: str = Field(default_factory=lambda: uuid4().hex)
    action: Literal["保持分开", "修改定义", "聚焦编码", "否决解释", "分析方向"]
    text: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    code_ids: list[StrictInt] = Field(default_factory=list)
    researcher: str = Field(min_length=1)
    date: str = Field(default_factory=recorded_now)
    confirmed: StrictBool = False

    @model_validator(mode="after")
    def check_target(self):
        if self.action == "保持分开" and len(set(self.code_ids)) < 2:
            raise ValueError("保持分开需要至少两条编码")
        if self.action in ("修改定义", "聚焦编码") and not self.code_ids:
            raise ValueError("请指定要修改或聚焦的编码")
        if not all(value.strip() for value in (self.text, self.reason, self.researcher)):
            raise ValueError("决定、理由及研究者不能为空")
        return self


class ComparisonRecord(ResearchRecord):
    comparison_id: str = Field(default_factory=lambda: uuid4().hex)
    code_ids: list[StrictInt] = Field(min_length=2)
    similarities: str = ""
    differences: str = Field(min_length=1)
    conditions: str = ""
    implication: str = Field(min_length=1)
    open_question: str = ""
    researcher: str = Field(min_length=1)
    date: str = Field(default_factory=recorded_now)

    @model_validator(mode="after")
    def distinct_evidence(self):
        if len(set(self.code_ids)) < 2:
            raise ValueError("比较需要至少两条不同的证据")
        return self


class CategoryRecord(ResearchRecord):
    category_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    code_ids: list[StrictInt] = Field(default_factory=list)
    active: bool = True
    definition: str = ""
    properties: str = ""
    conditions: str = ""
    actions: str = ""
    consequences: str = ""
    boundaries: str = ""
    gaps: str = ""
    memo: str = ""
    researcher: str = ""
    date: str = ""
    status: Literal["待发展", "需补充资料", "目前有较充分支持", "研究者判断饱和"] = "待发展"
    judgement: str = ""
    history: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def judgement_needs_reasons(self):
        if self.status != "待发展" and not (self.judgement.strip() and self.researcher.strip()):
            raise ValueError("充分性判断需要研究者与比较依据；分数不能代替判断")
        return self


class SamplingTask(ResearchRecord):
    task_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: Literal["一般核查", "理论抽样"] = "一般核查"
    category_id: str = ""
    gap: str = Field(min_length=1)
    alternatives: str = ""
    target: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    questions: str = ""
    expected_change: str = ""
    status: Literal["计划中", "已收集", "已分析"] = "计划中"
    outcome: str = ""
    researcher: str = Field(min_length=1)
    date: str = Field(default_factory=recorded_now)

    @model_validator(mode="after")
    def theoretical_purpose(self):
        if self.kind == "理论抽样" and not (self.category_id and self.expected_change.strip()):
            raise ValueError("理论抽样需要关联类属并说明会改变什么理解")
        if self.status == "已分析" and not self.outcome.strip():
            raise ValueError("已分析的任务需要记录分析后果")
        return self


class Relationship(ResearchRecord):
    from_category: str
    to_category: str
    description: str = Field(min_length=1)
    code_ids: list[StrictInt] = Field(min_length=1)
    boundary: str = ""
    researcher: str = Field(min_length=1)
    date: str = Field(default_factory=recorded_now)


class ArgumentRecord(ResearchRecord):
    claim: str = ""
    supporting_code_ids: list[StrictInt] = Field(default_factory=list)
    contradicting_code_ids: list[StrictInt] = Field(default_factory=list)
    alternatives: str = ""
    boundaries: str = ""
    counterexample_effect: str = ""
    researcher: str = ""
    date: str = ""
    relationships: list[Relationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def claim_needs_evidence(self):
        if self.claim and not (self.supporting_code_ids and self.researcher):
            raise ValueError("论断需要支持编码及研究者")
        return self


class StudyReview(ResearchRecord):
    credibility: str = ""
    originality: str = ""
    resonance: str = ""
    usefulness: str = ""
    researcher: str = ""
    date: str = ""


class ResearchWorkspace(ResearchRecord):
    schema_version: Literal[1] = 1
    project_id: str = Field(default_factory=lambda: uuid4().hex)
    round_id: str = ""
    round_number: int = Field(default=1, ge=1)
    rounds: list[dict] = Field(default_factory=list)
    sources: list[SourceMetadata] = Field(default_factory=list)
    decisions: list[AnalyticDecision] = Field(default_factory=list)
    submitted_decision_ids: list[str] = Field(default_factory=list)
    comparisons: list[ComparisonRecord] = Field(default_factory=list)
    focused_code_ids: list[StrictInt] = Field(default_factory=list)
    categories: list[CategoryRecord] = Field(default_factory=list)
    sampling_tasks: list[SamplingTask] = Field(default_factory=list)
    argument: ArgumentRecord = Field(default_factory=ArgumentRecord)
    review: StudyReview = Field(default_factory=StudyReview)
    changes: list[str] = Field(default_factory=list)


def validate_workspace(workspace, codes) -> ResearchWorkspace:
    if isinstance(workspace, ResearchWorkspace):
        workspace = workspace.model_dump()
    parsed = ResearchWorkspace.model_validate(workspace)
    allowed = {code["code_id"] for code in codes}
    categories = {item.category_id for item in parsed.categories}
    references = [parsed.focused_code_ids, parsed.argument.supporting_code_ids,
                  parsed.argument.contradicting_code_ids]
    references.extend(item.code_ids for item in [*parsed.decisions, *parsed.comparisons, *parsed.categories])
    for relation in parsed.argument.relationships:
        references.append(relation.code_ids)
        if relation.from_category not in categories or relation.to_category not in categories:
            raise ValueError("关系引用了不存在的类属")
    if any(code_id not in allowed for ids in references for code_id in ids):
        raise ValueError("研究记录引用了不存在的编码")
    if any(task.category_id and task.category_id not in categories for task in parsed.sampling_tasks):
        raise ValueError("抽样任务引用了不存在的类属")
    decision_ids = {item.decision_id for item in parsed.decisions if item.confirmed}
    if not set(parsed.submitted_decision_ids).issubset(decision_ids):
        raise ValueError("提交的分析决定尚未确认或不存在")
    return parsed


def decisions_for_model(decisions) -> tuple[str, ...]:
    visible = []
    for value in decisions or []:
        decision = AnalyticDecision.model_validate(value)
        if decision.confirmed:
            visible.append(f"{decision.action}；编码 {decision.code_ids}；{decision.text}；理由：{decision.reason}")
    return tuple(visible)


def continuation_context(workspace: ResearchWorkspace) -> str:
    """Only analytic records; never copy arbitrary fields or private memo text."""
    import json

    return json.dumps({
        "focused_code_ids": workspace.focused_code_ids,
        "comparisons": [item.model_dump(exclude={"researcher", "date"}) for item in workspace.comparisons],
        "categories": [item.model_dump(exclude={"history", "researcher", "date", "judgement", "status"})
                       for item in workspace.categories if item.active],
        "sampling_tasks": [item.model_dump(exclude={"researcher", "date"}) for item in workspace.sampling_tasks],
    }, ensure_ascii=False)


def prepare_previous_result(result: dict) -> tuple[list[dict], ResearchWorkspace]:
    """Validate imported data and strip fields outside the code/workspace schema."""
    from agents.generation_agent import Code

    if not isinstance(result, dict) or not isinstance(result.get("codes"), list) or not result["codes"]:
        raise ValueError("请选择包含编码的 TAMA 完整结果 JSON")
    codes = [Code.model_validate({"source_chunks": [], **value}).model_dump() for value in result["codes"]]
    ids = [item["code_id"] for item in codes]
    if len(ids) != len(set(ids)) or any(code_id < 0 for code_id in ids):
        raise ValueError("前一轮编码编号重复或无效")
    allowed = set(ids)
    by_id = {item["code_id"]: item for item in codes}
    for code in codes:
        if code["merged_into"] is not None and code["merged_into"] not in allowed:
            raise ValueError("前一轮归并来源不存在")
        if not set(code["merged_from"]).issubset(allowed):
            raise ValueError("前一轮归并来源不存在")
        if any(by_id[child_id]["merged_into"] != code["code_id"] or child_id == code["code_id"]
               for child_id in code["merged_from"]):
            raise ValueError("前一轮归并关系前后不一致")
        cursor, visited = code, set()
        while cursor["merged_into"] is not None:
            if cursor["code_id"] in visited:
                raise ValueError("前一轮归并来源存在循环")
            visited.add(cursor["code_id"])
            cursor = by_id[cursor["merged_into"]]
        if code["merged_into"] is not None and code["code_id"] not in by_id[code["merged_into"]]["merged_from"]:
            raise ValueError("前一轮归并关系前后不一致")
        code["evidence_id"] = code["evidence_id"] or f"{result.get('session_name') or 'imported'}:{code['code_id']}"
        if not code["source"]["source_id"]:
            code["source"]["source_id"] = result.get("session_name") or "imported"
    workspace = validate_workspace(result.get("research_workspace") or {}, codes)
    if not result.get("research_workspace"):
        workspace.project_id = uuid5(NAMESPACE_URL, f"tama:{result.get('session_name') or 'imported'}:{ids}").hex
    for item in codes:
        if not any(source.source_id == item["source"]["source_id"] for source in workspace.sources):
            workspace.sources.append(SourceMetadata.model_validate(item["source"]))
    if not workspace.rounds:
        workspace.round_id = result.get("session_name") or "imported"
        workspace.rounds = [{"round_id": workspace.round_id, "date": result.get("timestamp") or ""}]
    return codes, workspace


def apply_decisions(codes, decisions) -> None:
    from codebook import revise_definition

    by_id = {code.code_id: code for code in codes}
    for raw in decisions or []:
        decision = AnalyticDecision.model_validate(raw)
        if not decision.confirmed:
            continue
        if any(code_id not in by_id for code_id in decision.code_ids):
            raise ValueError("分析决定引用了不存在的编码")
        if decision.action == "保持分开":
            # Reopen the complete affected family so retained children are not lost.
            affected = set(decision.code_ids)
            for code_id in list(affected):
                code = by_id[code_id]
                parent = by_id.get(code.merged_into, code)
                affected.update([parent.code_id, *parent.merged_from])
            for code_id in affected:
                code = by_id[code_id]
                code.merged_into = None
                code.merged_from = []
            for code_id in decision.code_ids:
                code = by_id[code_id]
                code.separate_from = list(dict.fromkeys([
                    *code.separate_from, *(other for other in decision.code_ids if other != code_id),
                ]))
        elif decision.action == "修改定义":
            for code_id in decision.code_ids:
                if by_id[code_id].merged_into is not None:
                    parent = by_id[by_id[code_id].merged_into]
                    family = [parent.code_id, *parent.merged_from]
                    for member_id in family:
                        by_id[member_id].merged_into = None
                        by_id[member_id].merged_from = []
                revise_definition(by_id[code_id], {"definition": decision.text}, decision.reason,
                                  author=decision.researcher)
                by_id[code_id].definition_locked = True
                for member_id in by_id[code_id].merged_from:
                    by_id[member_id].definition_review = "需复核"
                    by_id[member_id].review_note = "研究者修改了规范定义，需要回查这条原始摘录。"


def build_workspace(codes, code_map, round_id, source, previous=None, decisions=None) -> dict:
    workspace = deepcopy(previous) if previous is not None else ResearchWorkspace()
    workspace.round_number = len(workspace.rounds) + 1
    workspace.round_id = round_id
    workspace.rounds.append({"round_id": round_id, "date": recorded_now(), "source_id": source.source_id})
    if not any(item.source_id == source.source_id for item in workspace.sources):
        workspace.sources.append(source)
    prior_names = {item.name: item for item in workspace.categories}
    old_active = {item.category_id: item.code_ids[:] for item in workspace.categories if item.active}
    for item in workspace.categories:
        item.active = False
        if previous is not None:
            item.history.append({"date": recorded_now(), "status": item.status,
                                 "judgement": item.judgement, "code_ids": item.code_ids[:]})
            item.status, item.judgement = "待发展", ""
    groups = next((step.get("groups", []) for step in (code_map or {}).get("iterations", [])
                   if step.get("step") == 2), [])
    changes = []
    for group in groups:
        item = prior_names.get(group["name"])
        if item is None:
            item = CategoryRecord(name=group["name"], code_ids=group["code_ids"], memo=group.get("memo") or "")
            workspace.categories.append(item)
            changes.append(f"新增候选类属：{item.name}")
        elif set(item.code_ids) != set(group["code_ids"]):
            changes.append(f"类属「{item.name}」的关联编码发生变化，需要回查属性和边界")
        item.active, item.code_ids = True, list(group["code_ids"])
    for item in workspace.categories:
        if item.category_id in old_active and not item.active:
            changes.append(f"原类属「{item.name}」未进入本轮映射，旧记录保留待核对")
    for raw in decisions or []:
        decision = AnalyticDecision.model_validate(raw)
        workspace.decisions = [item for item in workspace.decisions if item.decision_id != decision.decision_id]
        workspace.decisions.append(decision)
        if decision.confirmed and decision.action == "聚焦编码":
            workspace.focused_code_ids = list(dict.fromkeys([*workspace.focused_code_ids, *decision.code_ids]))
    workspace.submitted_decision_ids = [item.decision_id for item in map(AnalyticDecision.model_validate, decisions or [])
                                       if item.confirmed]
    new_codes = [code for code in codes if (code.get("source") or {}).get("source_id") == source.source_id]
    changes.insert(0, f"本轮加入 {len(new_codes)} 条初始编码；新旧资料共同参与归并与比较")
    if previous is not None:
        changes.append("新资料已加入，先前的充分性判断保留在历史中，本轮需重新判断")
        if workspace.argument.claim:
            workspace.argument = ArgumentRecord()
            changes.append("先前论断保留在前轮结果，本轮论断留待研究者重新确认")
        workspace.review = StudyReview()
        if workspace.argument.relationships:
            workspace.argument.relationships = []
            changes.append("类属关系留待新旧资料比较后重新确认，前轮关系保留在前轮结果")
    workspace.changes = changes
    return validate_workspace(workspace, codes).model_dump()


def render_research_records(workspace) -> str:
    parsed = ResearchWorkspace.model_validate(workspace)
    lines = ["# 持续研究记录", "", f"项目：{parsed.project_id}；第 {parsed.round_number} 轮", "",
             "主题评分与理论充分性分别记录；以下研究判断来自研究者。", ""]
    lines.extend(f"- {change}" for change in parsed.changes)
    for source in parsed.sources:
        lines.extend(["", f"## 来源 {source.source_id}", ""])
        for field, label in (("data_type", "资料类型"), ("case_id", "匿名案例"), ("participant_id", "匿名参与者"),
                             ("recorded_at", "资料产生时间"), ("setting", "场景"), ("event_id", "事件")):
            lines.append(f"{label}：{getattr(source, field) or '未知'}")
    for comparison in parsed.comparisons:
        lines.extend(["", f"## 比较 {comparison.code_ids}", "",
                      f"相同：{comparison.similarities}", f"不同：{comparison.differences}",
                      f"条件：{comparison.conditions}", f"分析后果：{comparison.implication}",
                      f"未决问题：{comparison.open_question}",
                      f"研究者：{comparison.researcher}；记录时间：{comparison.date}"])
    lines.extend(["", "## 聚焦编码", "", str(parsed.focused_code_ids)])
    for category in parsed.categories:
        lines.extend(["", f"## 类属：{category.name}", "",
                      f"编码：{category.code_ids}；{'本轮候选' if category.active else '历史记录，待核对'}"])
        for field, label in (("definition", "定义"), ("properties", "属性与维度"), ("conditions", "条件"),
                             ("actions", "行动与互动"), ("consequences", "结果"), ("boundaries", "边界"),
                             ("gaps", "缺口"), ("memo", "备忘录"), ("status", "发展状况"),
                             ("judgement", "比较与判断依据"), ("researcher", "研究者"), ("date", "记录时间")):
            lines.append(f"{label}：{getattr(category, field) or '（尚未填写）'}")
    for decision in parsed.decisions:
        submitted = decision.decision_id in parsed.submitted_decision_ids
        lines.extend(["", f"## 分析决定：{decision.action}", "", decision.text,
                      f"理由：{decision.reason}；编码：{decision.code_ids}",
                      f"研究者：{decision.researcher}；时间：{decision.date}；本轮{'已提交' if submitted else '未提交给模型'}"])
    names = {item.category_id: item.name for item in parsed.categories}
    for task in parsed.sampling_tasks:
        lines.extend(["", f"## {task.kind}：{task.target}", "", f"类属：{names.get(task.category_id) or '未关联'}",
                      f"缺口：{task.gap}", f"不同解释：{task.alternatives}", f"选择理由：{task.rationale}",
                      f"开放追问：{task.questions}", f"预期改变：{task.expected_change}",
                      f"状态：{task.status}；分析后果：{task.outcome}",
                      f"研究者：{task.researcher}；时间：{task.date}"])
    argument = parsed.argument
    lines.extend(["", "## 整体论证", "", argument.claim or "（待研究者形成论断）",
                  f"支持编码：{argument.supporting_code_ids}", f"反证编码：{argument.contradicting_code_ids}",
                  f"不同解释：{argument.alternatives}", f"适用边界：{argument.boundaries}",
                  f"反例对解释的影响：{argument.counterexample_effect}",
                  f"研究者：{argument.researcher}；时间：{argument.date}"])
    for relation in argument.relationships:
        lines.append(f"- {names[relation.from_category]} → {names[relation.to_category]}：{relation.description}；"
                     f"证据 {relation.code_ids}；边界：{relation.boundary}")
    lines.extend(["", "## 研究评议", ""])
    for field, label in (("credibility", "可信性"), ("originality", "原创性"),
                         ("resonance", "共鸣"), ("usefulness", "有用性")):
        lines.append(f"{label}：{getattr(parsed.review, field) or '（尚未评议；不能由模型分数推定）'}")
    lines.append(f"研究者：{parsed.review.researcher}；时间：{parsed.review.date}")
    return "\n".join(lines).rstrip() + "\n"
