"""Small Streamlit editors for researcher-owned analytic records."""

import json
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from next_data_plan import build_next_data_plan
from research_records import (
    AnalyticDecision, ArgumentRecord, CategoryRecord, ComparisonRecord, Relationship,
    SamplingTask, SourceMetadata, StudyReview, continuation_context, prepare_previous_result,
    recorded_now, render_research_records, validate_workspace,
)


def research_inputs(current_result, running):
    options, error = {}, None
    with st.expander("来源与持续研究（选填）"):
        mode = st.selectbox("分析方式", ["主题分析", "建构扎根理论支持"], key="analysis_mode", disabled=running)
        if mode == "建构扎根理论支持":
            options["analysis_mode"] = "grounded_theory"
            st.caption("记录比较、类属发展与抽样目的；充分性由研究者判断。")
        values = {}
        for field, label in (("source_id", "本份资料编号"), ("participant_id", "参与者匿名编号（单一参与者时填写）"),
                             ("recorded_at", "资料产生时间"), ("setting", "场景或地点"), ("event_id", "事件编号")):
            values[field] = st.text_input(label, key=f"source_{field}", disabled=running).strip()
        values["data_type"] = st.selectbox("资料类型", ["访谈", "观察笔记", "已有文本", "其他文字材料"],
                                          key="source_data_type", disabled=running)
        st.caption("每份资料使用独立编号；编号留空时自动生成。时间、地点、参与者留空表示未知。")
        st.caption("资料包含多个参与者时可留空参与者编号，逐条证据另显示材料中明确的说话人标签。")
        if any(value for key, value in values.items() if key != "data_type") or values["data_type"] != "访谈":
            options["source_metadata"] = SourceMetadata(**values).model_dump()
        usable = bool(current_result and current_result.get("codes"))
        use_current = st.checkbox("接续当前结果，加入本轮新资料", key="continue_current_research",
                                  disabled=running or not usable)
        uploaded = st.file_uploader("或载入前一轮完整结果 JSON", type=["json"], key="previous_research_json", disabled=running)
        previous = current_result if use_current and usable else None
        if uploaded is not None:
            if previous is not None:
                error = "当前结果与上传结果请只选择一个作为前轮资料。"
            else:
                try:
                    previous = json.loads(uploaded.getvalue().decode("utf-8-sig"))
                except (ValueError, UnicodeDecodeError):
                    error = "前轮结果无法读取，请使用保存完整数据得到的 JSON。"
        if previous is not None and error is None:
            try:
                codes, workspace = prepare_previous_result(previous)
            except (ValueError, TypeError, KeyError):
                error = "前轮结果格式或证据编号无效，请使用完整的 TAMA 结果。"
            else:
                options["previous_result"] = previous
                scope = f"{workspace.project_id}:{workspace.round_id}"
                if st.session_state.get("_previous_research_scope") != scope:
                    for key in ("submitted_analysis_decisions", "confirm_submit_decisions", "include_previous_context"):
                        st.session_state.pop(key, None)
                    st.session_state["_previous_research_scope"] = scope
                st.caption(f"已载入 {len(codes)} 条编码；新资料将进入第 {len(workspace.rounds) + 1} 轮。")
                context = st.checkbox("将前轮类属、比较与抽样记录用于本轮分析", key="include_previous_context", disabled=running)
                if context:
                    options["include_previous_context"] = True
                    st.caption("下面显示本轮会提交的分析记录；私人手记和自反记录不在其中。")
                    st.json(json.loads(continuation_context(workspace)), expanded=False)
                confirmed = {item.decision_id: item for item in workspace.decisions if item.confirmed}
                selected = st.multiselect("本轮要提交的已确认分析决定", list(confirmed),
                                          format_func=lambda key: f"{confirmed[key].action}：{confirmed[key].text}",
                                          key="submitted_analysis_decisions", disabled=running,
                                          on_change=lambda: st.session_state.update(confirm_submit_decisions=False))
                for key in selected:
                    st.write(f"{confirmed[key].text}；编码 {confirmed[key].code_ids}；理由：{confirmed[key].reason}")
                if selected:
                    consent = st.checkbox("确认将上面选定的决定用于本轮模型分析", key="confirm_submit_decisions", disabled=running)
                    if consent:
                        options["confirmed_decisions"] = [confirmed[key].model_dump() for key in selected]
                    else:
                        error = "选定的分析决定尚未确认提交；请确认或取消选择。"
        if error:
            st.warning(error)
    return options, error


def save_workspace(result, workspace, root):
    parsed = validate_workspace(workspace, result.get("codes") or [])
    result["research_workspace"] = parsed.model_dump()
    result["next_data_plan"] = build_next_data_plan(result.get("codes"),
                                                    [*(result.get("final_themes") or []),
                                                     *(item["theme"] for item in result.get("retired_findings") or [])],
                                                    result.get("memos"), result["research_workspace"])
    if result.get("configuration", {}).get("save_final") and result.get("output_dir"):
        # Imported output paths never control writes; only this app's output tree is writable here.
        expected = (Path(root) / "outputs" / str(result.get("session_name") or "")).resolve()
        base = (Path(root) / "outputs").resolve()
        target = Path(result["output_dir"]).resolve()
        if target != expected or not target.is_relative_to(base) or not target.is_dir():
            raise ValueError("本地保存位置与本轮输出不一致，请下载完整数据保存。")
        from reporting import render_summary_text
        from codebook import render_codebook

        files = {
            "00_final_results.json": json.dumps(result, ensure_ascii=False, indent=2),
            "08_research_records.md": render_research_records(parsed),
            "next_data_plan.md": result["next_data_plan"]["markdown"],
            "00_summary.txt": render_summary_text(result),
            "05_codebook.md": render_codebook(result.get("codes") or []),
        }
        for name, content in files.items():
            pending = target / f".{name}.tmp"
            pending.write_text(content, encoding="utf-8")
            pending.replace(target / name)


def _save(result, workspace, root):
    try:
        save_workspace(result, workspace, root)
    except (ValueError, OSError):
        st.error("记录未完整保存，请核对编码、类属及本地保存位置。")
    else:
        st.session_state["research_record_notice"] = "研究记录已更新，可下载完整数据；未发送模型请求。"
        st.rerun()


def _code_picker(label, codes, key, default=None):
    by_id = {code["code_id"]: code for code in codes}
    return st.multiselect(label, list(by_id), default=[value for value in default or [] if value in by_id],
                          format_func=lambda value: f"[{value}] {by_id[value].get('name') or by_id[value]['description']}", key=key)


def _comparison_editor(result, workspace, root):
    codes = result.get("codes") or []
    with st.expander("比较原话、情境与定义"):
        selected = _code_picker("选择要比较的证据", codes, "comparison_code_ids")
        by_id = {code["code_id"]: code for code in codes}
        for code_id in selected:
            code = by_id[code_id]
            st.markdown(f"**[{code_id}] {code.get('name') or code['description']}**")
            source = code.get("source") or {}
            st.caption(f"来源 {source.get('source_id') or '未知'} · 参与者 {source.get('participant_id') or '未知'} · "
                       f"说话人 {code.get('speaker') or '未知'} · 时间 {source.get('recorded_at') or '未知'} · "
                       f"事件 {source.get('event_id') or '未知'} · 场景 {source.get('setting') or '未知'}")
            st.write(code.get("excerpt") or "没有可核对的逐字摘录")
            st.text(code.get("context") or "没有相邻语境")
            canonical = by_id.get(code.get("merged_into"), code)
            st.write(f"当前定义：{canonical.get('definition') or '尚未定义'}")
            st.caption(f"定义复核：{code.get('definition_review') or '未复核'}；{code.get('review_note') or ''}")
            if code.get("definition_history"):
                st.json(code["definition_history"], expanded=False)
        with st.form("comparison_form"):
            fields = {}
            for field, label in (("similarities", "相同之处"), ("differences", "不同之处"),
                                 ("conditions", "与差异有关的条件"), ("implication", "这次比较改变了什么理解"),
                                 ("open_question", "接下来要追问什么")):
                fields[field] = st.text_area(label, key=f"comparison_{field}")
            researcher = st.text_input("记录研究者（可用代号）", key="comparison_researcher")
            if st.form_submit_button("保存比较记录"):
                try:
                    comparison = ComparisonRecord(code_ids=selected, researcher=researcher, **fields)
                except ValidationError:
                    st.error("请选择至少两条不同证据，并填写差异、分析后果及研究者。")
                else:
                    workspace.comparisons.append(comparison)
                    _save(result, workspace, root)
        if selected:
            with st.form("definition_review_form"):
                target = st.selectbox("要回查的证据", selected, key="definition_review_code")
                status = st.selectbox("新定义是否适用于这条原话", ["需复核", "适用", "不适用"], key="definition_review_status")
                reason = st.text_area("复核理由", key="definition_review_reason")
                author = st.text_input("复核研究者", key="definition_review_researcher")
                if st.form_submit_button("保存定义复核"):
                    if not reason.strip() or not author.strip():
                        st.error("请写下复核理由及研究者。")
                    else:
                        by_id[target]["definition_review"] = status
                        by_id[target]["review_note"] = f"{author.strip()} · {recorded_now()} · {reason.strip()}"
                        _save(result, workspace, root)


def _decision_editor(result, workspace, root):
    with st.expander("记录分析决定与聚焦代码"):
        st.caption("这里只保存决定；在下一轮选中并确认提交后才进入模型。私人手记仍不提交。")
        with st.form("analysis_decision_form"):
            action = st.selectbox("决定类型", ["保持分开", "修改定义", "聚焦编码", "否决解释", "分析方向"], key="decision_action")
            ids = _code_picker("关联编码", result.get("codes") or [], "decision_code_ids")
            text = st.text_area("具体决定（修改定义时填写新定义）", key="decision_text")
            reason = st.text_area("决定的证据与理由", key="decision_reason")
            researcher = st.text_input("决定研究者", key="decision_researcher")
            confirmed = st.checkbox("我已核对并确认这项决定", key="decision_confirmed")
            if st.form_submit_button("保存分析决定"):
                try:
                    decision = AnalyticDecision(action=action, text=text, reason=reason, code_ids=ids,
                                                researcher=researcher, confirmed=confirmed)
                except ValidationError:
                    st.error("请填写决定、理由和研究者；分开至少指定两条编码，定义或聚焦需指定编码。")
                else:
                    workspace.decisions.append(decision)
                    if confirmed and action == "聚焦编码":
                        workspace.focused_code_ids = list(dict.fromkeys([*workspace.focused_code_ids, *ids]))
                    _save(result, workspace, root)
        for decision in workspace.decisions:
            st.write(f"{'已确认' if decision.confirmed else '草稿'} · {decision.action} · {decision.text} · {decision.code_ids}")
        st.caption(f"研究者选定的聚焦编码：{workspace.focused_code_ids or '尚未选择'}")


def _category_editor(result, workspace, root):
    with st.expander("发展类属与记录充分性判断"):
        st.caption("这些分析记录先保存在本轮结果；下一轮勾选使用前轮分析记录时才会提交。私人反思请写在自反栏。")
        categories = {item.category_id: item for item in workspace.categories}
        if not categories:
            st.info("本轮尚无候选类属；可先记录比较与聚焦代码。")
            return
        chosen = st.selectbox("类属", list(categories), format_func=lambda key: categories[key].name +
                             ("" if categories[key].active else "（历史记录）"), key="category_editor_id")
        original = categories[chosen]
        with st.form(f"category_form_{chosen}"):
            fields = {}
            for field, label in (("definition", "暂定定义"), ("properties", "属性与变化维度"),
                                 ("conditions", "发生、保持或改变的条件"), ("actions", "行动与互动"),
                                 ("consequences", "结果"), ("boundaries", "与相邻类属的区别和边界"),
                                 ("gaps", "资料或解释缺口"), ("memo", "自由分析备忘录")):
                fields[field] = st.text_area(label, value=getattr(original, field), key=f"category_{chosen}_{field}")
            fields["code_ids"] = _code_picker("检验该类属的编码", result.get("codes") or [], f"category_{chosen}_codes", original.code_ids)
            statuses = ["待发展", "需补充资料", "目前有较充分支持", "研究者判断饱和"]
            fields["status"] = st.selectbox("类属发展状况", statuses, index=statuses.index(original.status), key=f"category_{chosen}_status")
            fields["judgement"] = st.text_area("比较依据与停止或继续的理由", value=original.judgement, key=f"category_{chosen}_judgement")
            fields["researcher"] = st.text_input("判断研究者", value=original.researcher, key=f"category_{chosen}_researcher")
            st.caption("没有新主题、分数达标和运行停止都不能单独证明饱和。上面的分析问题允许留空。")
            if st.form_submit_button("保存类属发展记录"):
                try:
                    updated = CategoryRecord.model_validate({**original.model_dump(), **fields, "date": recorded_now()})
                except ValidationError:
                    st.error("请为发展状况提供研究者和比较依据。")
                else:
                    updated.history.append({"date": original.date, "before": original.model_dump(exclude={"history"})})
                    workspace.categories = [updated if item.category_id == chosen else item for item in workspace.categories]
                    _save(result, workspace, root)
        if original.history:
            st.json(original.history, expanded=False)


def _sampling_editor(result, workspace, root):
    with st.expander("计划补充资料并回填分析后果"):
        existing = {item.task_id: item for item in workspace.sampling_tasks}
        selected = st.selectbox("资料任务", ["新任务", *existing],
                                format_func=lambda key: key if key == "新任务" else existing[key].target,
                                key="sampling_task_selection")
        previous = existing.get(selected)
        defaults = previous.model_dump() if previous else {}
        categories = {item.category_id: item for item in workspace.categories}
        with st.form(f"sampling_form_{selected}"):
            kinds = ["一般核查", "理论抽样"]
            fields = {"kind": st.selectbox("任务目的", kinds, index=kinds.index(defaults.get("kind", "一般核查")), key=f"task_{selected}_kind")}
            category_options = ["", *categories]
            fields["category_id"] = st.selectbox("要发展的类属", category_options,
                index=category_options.index(defaults.get("category_id", "")),
                format_func=lambda key: categories[key].name if key else "未关联（一般核查可留空）", key=f"task_{selected}_category")
            for field, label in (("gap", "要补充的属性、关系或事实缺口"), ("alternatives", "需要区分的不同解释"),
                                 ("target", "要寻找的人、事件或资料"), ("rationale", "为什么选择这些资料"),
                                 ("questions", "开放追问"), ("expected_change", "哪些新观察会改变现有理解"),
                                 ("outcome", "收集后得到什么，怎样修改分析")):
                fields[field] = st.text_area(label, value=defaults.get(field, ""), key=f"task_{selected}_{field}")
            statuses = ["计划中", "已收集", "已分析"]
            fields["status"] = st.selectbox("任务进展", statuses, index=statuses.index(defaults.get("status", "计划中")), key=f"task_{selected}_status")
            fields["researcher"] = st.text_input("任务研究者", value=defaults.get("researcher", ""), key=f"task_{selected}_researcher")
            if st.form_submit_button("保存资料任务"):
                try:
                    task = SamplingTask.model_validate({**defaults, **fields, "date": recorded_now()})
                except ValidationError:
                    st.error("请填写缺口、目标、选择理由和研究者；理论抽样需关联类属与预期改变，已分析需回填后果。")
                else:
                    workspace.sampling_tasks = [item for item in workspace.sampling_tasks if item.task_id != task.task_id] + [task]
                    _save(result, workspace, root)


def _argument_editor(result, workspace, root):
    with st.expander("整体论证、类属关系与研究评议"):
        with st.form("research_argument_form"):
            fields = {}
            for field, label in (("claim", "这一轮的整体论断"), ("alternatives", "仍可能成立的不同解释"),
                                 ("boundaries", "解释适用的条件与边界"), ("counterexample_effect", "反例怎样改变了解释")):
                fields[field] = st.text_area(label, value=getattr(workspace.argument, field), key=f"argument_{field}")
            for field, label in (("supporting_code_ids", "支持论断的编码"), ("contradicting_code_ids", "不支持论断的编码")):
                fields[field] = _code_picker(label, result.get("codes") or [], f"argument_{field}", getattr(workspace.argument, field))
            fields["researcher"] = st.text_input("论断研究者", value=workspace.argument.researcher, key="argument_researcher")
            review = {}
            for field, label in (("credibility", "可信性：比较、资料深度与证据联系"), ("originality", "原创性：与真实文献的对话及贡献"),
                                 ("resonance", "共鸣：参与者或同背景者的反馈"), ("usefulness", "有用性：实践或知识贡献")):
                review[field] = st.text_area(label, value=getattr(workspace.review, field), key=f"review_{field}")
            if st.form_submit_button("保存论证与研究评议"):
                if (any(fields[field].strip() for field in ("claim", "alternatives", "boundaries", "counterexample_effect"))
                        or any(value.strip() for value in review.values())) and not fields["researcher"].strip():
                    st.error("请填写论断研究者。")
                else:
                    try:
                        workspace.argument = ArgumentRecord(**fields, date=recorded_now(), relationships=workspace.argument.relationships)
                    except ValidationError:
                        st.error("整体论断需要指定支持编码与研究者。")
                    else:
                        workspace.review = StudyReview(**review, researcher=fields["researcher"], date=recorded_now())
                        _save(result, workspace, root)
        categories = {item.category_id: item for item in workspace.categories if item.active}
        if len(categories) >= 2:
            with st.form("category_relationship_form"):
                start = st.selectbox("关系起点", list(categories), format_func=lambda key: categories[key].name, key="relation_start")
                end = st.selectbox("关系终点", list(categories), index=1, format_func=lambda key: categories[key].name, key="relation_end")
                description = st.text_area("两者如何关联（允许并行、反复；不推定因果）", key="relation_description")
                ids = _code_picker("支持这个关系的编码", result.get("codes") or [], "relation_codes")
                boundary = st.text_area("关系的条件与边界", key="relation_boundary")
                researcher = st.text_input("关系研究者", key="relation_researcher")
                if st.form_submit_button("保存类属关系"):
                    try:
                        if start == end:
                            raise ValueError("请比较不同类属")
                        relation = Relationship(from_category=start, to_category=end, description=description, code_ids=ids,
                                                boundary=boundary, researcher=researcher)
                    except ValueError:
                        st.error("请选择不同的类属，并填写关系、真实编码证据和研究者。")
                    else:
                        workspace.argument.relationships.append(relation)
                        _save(result, workspace, root)
        if workspace.argument.relationships:
            all_names = {item.category_id: item.name for item in workspace.categories}
            for item in workspace.argument.relationships:
                st.write(f"{all_names[item.from_category]} → {all_names[item.to_category]}：{item.description}；证据 {item.code_ids}")
            st.caption("箭头仅表达研究者记录的关联，具体含义与边界见文字。")
            st.graphviz_chart(_relationship_dot(workspace))


def _relationship_dot(workspace):
    names = {item.category_id: item.name for item in workspace.categories}
    lines = ["digraph relations {"]
    for item in workspace.argument.relationships:
        lines.append(f"{json.dumps(item.from_category)} -> {json.dumps(item.to_category)} "
                     f"[label={json.dumps(item.description, ensure_ascii=False)}];")
    for key, name in names.items():
        lines.append(f"{json.dumps(key)} [label={json.dumps(name, ensure_ascii=False)}];")
    return "\n".join([*lines, "}"])


def render_research_tools(result, root):
    if not result.get("research_workspace"):
        return
    workspace = validate_workspace(result["research_workspace"], result.get("codes") or [])
    if st.session_state.get("_research_editor_round") != workspace.round_id:
        prefixes = ("comparison_", "decision_", "category_", "task_", "argument_", "review_", "relation_", "definition_review_")
        for key in list(st.session_state):
            if (key.startswith(prefixes) and key != "decision_mode") or key == "sampling_task_selection":
                st.session_state.pop(key, None)
        st.session_state["_research_editor_round"] = workspace.round_id
    st.subheader("持续研究记录")
    st.caption(f"第 {workspace.round_number} 轮 · 新资料加入后可重新比较与修改解释；这些表单不会发送模型请求。")
    notice = st.session_state.pop("research_record_notice", None)
    if notice:
        st.success(notice)
    for change in workspace.changes:
        st.write(change)
    _comparison_editor(result, workspace, root)
    _decision_editor(result, workspace, root)
    if result.get("configuration", {}).get("analysis_mode") == "grounded_theory":
        _category_editor(result, workspace, root)
        _sampling_editor(result, workspace, root)
        _argument_editor(result, workspace, root)
