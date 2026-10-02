"""Local comparison, interpretation and review controls; no model calls."""

from copy import deepcopy
from uuid import uuid4

import streamlit as st

from analysis_records import REVIEW_LABELS, record_event, recorded_now, update_annotations
from evidence import matched_codes
from evidence_views import DIMENSIONS, build_comparison_matrix, evidence_context
from research_ui import save_result_records


def render_evidence(result, code, key):
    source = code.get("source") or {}
    st.write(f"[{code['code_id']}] {code.get('name') or code.get('description')}")
    st.write(f"原话：{code.get('excerpt') or '没有有效摘录'}")
    st.caption(f"来源 {source.get('source_id') or '未知'} · 参与者 {source.get('participant_id') or '未知'} · "
               f"说话人 {code.get('speaker') or '未知'} · 场景 {source.get('setting') or '未知'} · "
               f"时间 {source.get('recorded_at') or '未知'} · 原文位置 {code.get('source_start')}–{code.get('source_end')}")
    if st.checkbox("展开所属问答或段落", key=key):
        context = evidence_context(result, code)
        st.caption(f"{context['kind']} · 来源 {context['source_id'] or '未知'} · 区间 {context['start']}–{context['end']}")
        st.text(context["text"] or "没有可用语境")


def render_matrix(result):
    with st.expander("材料与主题比较"):
        dimension = st.selectbox("比较维度", list(DIMENSIONS), format_func=DIMENSIONS.get,
                                index=list(DIMENSIONS).index((result.get("report_options") or {}).get("matrix_dimension", "participant_id")), key="matrix_dimension")
        matrix = build_comparison_matrix(result, dimension)
        counts = matrix["counts"]
        st.caption(f"有效编码 {counts['codes']} 条 · 不同原文位置 {counts['evidence']} 处 · 材料 {counts['sources']} 份 · "
                   f"已标记参与者 {counts['participants']} 人 · 已标记案例 {counts['cases']} 个")
        st.caption(f"参与者未知的编码 {counts['unknown_participants']} 条；案例未知的编码 {counts['unknown_cases']} 条。"
                   "说话人标签不自动等同参与者；次数不代表重要性。“未涉及”仅指未关联到有效证据，不代表反对或刻意沉默。")
        if not matrix["rows"] or not result.get("final_themes"):
            st.info("暂无可比较的有效证据与主题。")
            return
        columns = {theme["theme_id"]: f"{index}. {theme['name']}" for index, theme in enumerate(result["final_themes"], 1)}
        st.dataframe([{DIMENSIONS[dimension]: row["label"],
                       **{columns[cell["theme_id"]]: cell["state"] for cell in row["cells"]}}
                      for row in matrix["rows"]], use_container_width=True, hide_index=True)
        row_index = st.selectbox("查看分组", range(len(matrix["rows"])),
                                 format_func=lambda index: matrix["rows"][index]["label"], key="matrix_row")
        theme_id = st.selectbox("查看主题证据", list(columns), format_func=columns.get, key="matrix_theme")
        cell = next(item for item in matrix["rows"][row_index]["cells"] if item["theme_id"] == theme_id)
        st.write(f"本格：{cell['state']}")
        by_id = {code["code_id"]: code for code in result.get("codes") or []}
        for role, ids in (("支持", cell["supporting_code_ids"]), ("反证", cell["contradicting_code_ids"])):
            for code_id in ids:
                st.caption(role)
                render_evidence(result, by_id[code_id], f"matrix_evidence_{row_index}_{theme_id}_{code_id}")
        _group_editor(result)


def _group_editor(result):
    codes = {code["code_id"]: code for code in matched_codes(result.get("codes") or [])}
    if not codes:
        return
    code_id = st.selectbox("确认某条证据的分组", list(codes), key="matrix_group_code")
    code = codes[code_id]
    st.caption("只填写原稿已有信息或你已确认的匿名编号；未知可以留空。这不修改原稿或发送模型请求。")
    with st.form(f"matrix_group_form_{code_id}"):
        fields = {}
        for key in ("participant_id", "case_id", "event_id", "recorded_at"):
            fields[key] = st.text_input(DIMENSIONS[key], value=(code.get("source") or {}).get(key, ""), key=f"matrix_group_{code_id}_{key}").strip()
        researcher = st.text_input("分组确认研究者", key=f"matrix_group_{code_id}_researcher")
        reason = st.text_area("分组依据", key=f"matrix_group_{code_id}_reason")
        if st.form_submit_button("保存证据分组"):
            if not researcher.strip() or not reason.strip():
                st.error("请填写分组确认研究者与依据。")
                return
            from pathlib import Path
            candidate = deepcopy(result)
            changed = next(item for item in candidate["codes"] if item["code_id"] == code_id)
            before = dict(changed.get("source") or {})
            changed["source"] = {**before, **fields}
            record_event(candidate.setdefault("audit_trail", []), "evidence_group", "研究者", "确认原文证据分组",
                         {"code_id": code_id, "before": before, "after": changed["source"], "researcher": researcher, "reason": reason})
            if ((candidate.get("researcher_annotations") or {}).get("review") or {}).get("status") == "reviewed":
                candidate["researcher_annotations"]["review"]["status"] = "pending"
            try:
                save_result_records(result, candidate, Path(__file__).resolve().parents[1])
            except (ValueError, OSError):
                st.error("分组未保存，请核对本地保存位置。")
            else:
                st.session_state["research_record_notice"] = "证据分组已保存，原稿未改变；未发送模型请求。"
                st.rerun()


def save_annotations_ui(result, root, annotations, include_context=None, confirm_review=False):
    try:
        candidate = update_annotations(result, annotations, include_context, confirm_review=confirm_review,
                                       matrix_dimension=st.session_state.get("matrix_dimension"))
        save_result_records(result, candidate, root)
    except (ValueError, OSError):
        st.error("记录未保存。请核对解释的证据编号、复核研究者与理由，以及本地保存位置。")
    else:
        st.session_state["research_record_notice"] = "研究者记录已保存到本次结果；可下载完整 JSON 和报告，未发送模型请求。"
        st.rerun()


def render_review(result, root):
    annotations = result.get("researcher_annotations") or {}
    review = annotations.get("review") or {}
    if result.get("annotations_need_review"):
        st.warning("原稿版本与诠释保存时不同，请重新核对研究者记录。")
    st.caption(f"研究者复核状态：{REVIEW_LABELS.get(review.get('status'), '尚未复核')}")
    with st.expander("解释比较与研究者复核"):
        st.caption("分别记录可能的解释、支持与反证及解释边界。保存不发送模型请求；写明取舍理由后才标记复核状态。")
        old = annotations.get("explanations") or []
        codes = {code["code_id"]: code for code in matched_codes(result.get("codes") or [])}
        ids = [old[index]["explanation_id"] if index < len(old) else uuid4().hex for index in range(2)]
        with st.form("explanation_review_form"):
            explanations = []
            for index in range(2):
                previous = old[index] if index < len(old) else {}
                prefix = f"explanation_{index}"
                text = st.text_area(f"解释 {index + 1}", value=previous.get("text", ""), key=f"{prefix}_text")
                supports = st.multiselect(f"解释 {index + 1} 的支持编码", list(codes),
                    default=[value for value in previous.get("supporting_code_ids", []) if value in codes],
                    format_func=lambda value: f"[{value}] {codes[value].get('name') or codes[value]['description']}", key=f"{prefix}_supports")
                counters = st.multiselect(f"解释 {index + 1} 的反证编码", list(codes),
                    default=[value for value in previous.get("contradicting_code_ids", []) if value in codes],
                    format_func=lambda value: f"[{value}] {codes[value].get('name') or codes[value]['description']}", key=f"{prefix}_counters")
                boundaries = st.text_area(f"解释 {index + 1} 的适用边界", value=previous.get("boundaries", ""), key=f"{prefix}_boundaries")
                unresolved = st.text_area(f"解释 {index + 1} 无法解释什么", value=previous.get("unresolved", ""), key=f"{prefix}_unresolved")
                if text.strip():
                    explanations.append({"explanation_id": ids[index], "text": text, "supporting_code_ids": supports,
                                         "contradicting_code_ids": counters, "boundaries": boundaries, "unresolved": unresolved})
            selected = 1 + ids.index(review["selected_explanation_id"]) if review.get("selected_explanation_id") in ids else 0
            choice = st.selectbox("本次采用哪种解释", ["暂不采用", "解释 1", "解释 2"], index=selected, key="explanation_choice")
            statuses = list(REVIEW_LABELS)
            status = st.selectbox("复核状态", statuses, index=statuses.index(review.get("status", "pending")),
                                  format_func=REVIEW_LABELS.get, key="explanation_status")
            researcher = st.text_input("复核研究者（可用代号）", value=review.get("researcher", ""), key="explanation_researcher")
            reason = st.text_area("保留、缩小或否决解释的理由", value=review.get("reason", ""), key="explanation_reason")
            include_context = st.checkbox("报告附选中引文的原文语境", value=(result.get("report_options") or {}).get("include_context", False), key="report_include_context")
            if st.form_submit_button("保存解释比较与复核"):
                updated = {**annotations, "explanations": explanations + old[2:],
                           "review": {"status": status, "researcher": researcher, "reason": reason,
                                      "selected_explanation_id": ids[int(choice[-1]) - 1] if choice != "暂不采用" else "",
                                      "date": recorded_now()}}
                save_annotations_ui(result, root, updated, include_context, confirm_review=True)
    with st.expander("分析与复核过程"):
        events = result.get("audit_trail") or []
        if not events:
            st.info("这份结果尚未保存过程事件；旧版结果不会被补造历史。")
        for index, event in enumerate(events, 1):
            st.write(f"{index}. {event['summary']} · {event['actor']} · {event['date']}")
            st.json(event.get("details") or {}, expanded=False)
        for finding in result.get("retired_findings") or []:
            st.write(f"保留的历史发现：{finding['theme']['name']}；修订理由：{finding.get('reason') or '未记录'}")
