"""Presentation helpers for the white, typography-led research workspace."""

from html import escape
from math import isfinite
from pathlib import Path

import streamlit as st

from evidence import is_matched
from reporting import KIND_LABELS
from analysis_progress import normalize_phase


def format_duration(seconds: float | None) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds < 0:
        return "—"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_job_progress(snapshot) -> None:
    """Show exact batch progress without inventing a whole-analysis estimate."""
    phase = normalize_phase(snapshot.stage)
    total = getattr(snapshot, "total", 0)
    completed = getattr(snapshot, "completed", 0)
    active = getattr(snapshot, "active", 0)
    limit = getattr(snapshot, "concurrency_limit", None)
    limit_label = str(limit) if limit is not None else "—"
    unit = getattr(snapshot, "unit", "任务")
    paused = snapshot.pause_requested
    state = "正在取消" if snapshot.cancelled else "已请求暂停" if paused else "分析进行中"
    modifier = " is-paused" if paused or snapshot.cancelled else ""
    with st.container(key="analysis_progress", border=True):
        st.markdown(
            f'<div class="tl-progress-heading{modifier}"><div>'
            f'<span class="tl-progress-state"><i aria-hidden="true"></i>{state}</span>'
            f'<h3>{escape(phase)}</h3></div>'
            '<span class="tl-progress-refresh">每秒更新</span></div>', unsafe_allow_html=True,
        )
        time_col, phase_col, active_col = st.columns(3)
        time_col.metric("累计用时", format_duration(getattr(snapshot, "elapsed_seconds", None)))
        phase_col.metric("当前步骤用时", format_duration(getattr(snapshot, "stage_elapsed_seconds", None)),
                         help="从当前批次或步骤开始计算。累计用时包含等待与暂停。")
        active_col.metric("正在处理 / 并发上限", f"{active} / {limit_label}" if total else f"— / {limit_label}")
        if not hasattr(snapshot, "elapsed_seconds"):
            st.caption("本次任务在更新前启动，无法补算累计用时；新任务会完整记录进度。")

        iteration = getattr(snapshot, "iteration", 0)
        if iteration:
            st.caption(f"第 {iteration} / {getattr(snapshot, 'max_iterations', iteration)} 轮评估与修订")
        if total > 1:
            st.progress(min(completed / total, 1.0), text=f"本批次已处理 {completed} / {total} 个{unit}")
            failed = getattr(snapshot, "failed", 0)
            remaining = max(total - completed - active, 0)
            st.caption(f"正在处理 {active} 个 · 等待处理 {remaining} 个" + (f" · 未成功 {failed} 个" if failed else ""))
        elif total == 1:
            if completed:
                st.caption("本步已完成，正在进入下一步。")
            elif phase not in {"切分材料", "校验全文回查证据", "整理本次结果"}:
                st.caption("正在等待本步骤的模型结果。")

        explanations = {
            "准备运行": "正在准备分析任务和模型连接。",
            "切分材料": "正在按自然语义边界整理访谈片段。",
            "提取编码": "多个访谈片段独立编码，完成一个就更新一次计数。",
            "归并编码": "正在合并重复编码；本步骤按顺序处理，提高并发对这一阶段帮助有限。",
            "映射编码": "正在将片段编码对应到统一编码，本步骤按顺序处理。",
            "归纳主题": "正在从编码中形成候选主题；独立批次并行，最终整合按顺序进行。",
            "合并候选主题": "正在归并候选主题；独立批次并行，最终整合按顺序进行。",
            "全文回查": "正在逐片检查遗漏和反例，完成一个片段就更新一次计数。",
            "校验全文回查证据": "正在核对回查发现与原文是否一致。",
            "评估主题": "正在检查各主题的质量；必要时还会等待模型生成改进建议。",
            "修订主题": "正在结合评估反馈调整主题，本步骤按顺序处理。",
            "整理本次结果": "正在汇总本次报告和记录。",
        }
        st.caption(explanations.get(phase, "正在处理当前步骤，收到结果后会更新。"))
        if getattr(snapshot, "idle_seconds", 0) >= 120 and not paused and not snapshot.cancelled:
            st.warning("超过 2 分钟没有新进展，可能仍在等待模型返回或接口重试；耗时还在统计。")

        events = getattr(snapshot, "recent_events", ())
        if events:
            st.markdown('<div class="tl-progress-label">最近进展</div><ol class="tl-progress-events">' + ''.join(
                f'<li><time>{format_duration(event.elapsed_seconds)}</time><span>{escape(event.message)}</span></li>'
                for event in reversed(events[-3:])
            ) + '</ol>', unsafe_allow_html=True)
        history = getattr(snapshot, "phase_history", ())
        if history:
            with st.expander(f"已进行的步骤 · {len(history)} 项"):
                labels = {"complete": "完成", "partial": "已结束", "cancelled": "取消", "error": "未完成"}
                for item in history:
                    count = f" · {item.completed}/{item.total} 个{item.unit}" if item.total else ""
                    round_label = f" · 第 {item.iteration} 轮" if item.iteration else ""
                    st.caption(f"{item.phase}{round_label} · {labels.get(item.state, '已结束')} · {format_duration(item.elapsed_seconds)}{count}")


def load_styles() -> None:
    css = Path(__file__).with_name("ui.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def workspace_header(*, compact: bool = False) -> None:
    if compact:
        return
    st.title("访谈质性分析")
    st.write("从访谈原文到编码、主题与诠释，在同一个工作台中阅读、比较和复核。")
    st.markdown(
        '<ol class="tl-workflow" aria-label="分析流程">'
        '<li><span>01</span>整理材料</li><li><span>02</span>分析与编码</li>'
        '<li><span>03</span>阅读与复核</li></ol>',
        unsafe_allow_html=True,
    )


def report_identity(result: dict) -> None:
    details = list(dict.fromkeys(str(value) for value in
                               (result.get("case_id"), result.get("session_name")) if value))
    if result.get("timestamp"):
        details.append(str(result["timestamp"]))
    duration = (result.get("metadata") or {}).get("elapsed_seconds")
    if format_duration(duration) != "—":
        details.append(f"总用时 {format_duration(duration)}")
    st.markdown(
        '<div class="tl-report-meta"><span class="tl-eyebrow">ANALYSIS REPORT</span>'
        f'<span>{escape(" · ".join(details))}</span></div>', unsafe_allow_html=True,
    )


def review_brief(result: dict) -> None:
    themes = result.get("final_themes") or []
    questions = {str(question) for theme in themes for question in theme.get("open_questions") or []}
    pending = sum(not is_matched(code) for code in result.get("codes") or [])
    review = ((result.get("researcher_annotations") or {}).get("review") or {}).get("status")
    from analysis_records import REVIEW_LABELS
    rows = [("研究者复核", REVIEW_LABELS.get(review, "尚未复核")),
            ("主题待核查问题", f"{len(questions)} 项"), ("待匹配原文的编码", f"{pending} 条")]
    st.markdown('<aside class="tl-review-brief"><div class="tl-eyebrow">阅读这份报告</div>'
                '<h3>需要确认的内容</h3>' + ''.join(
                    f'<div class="tl-brief-row"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
                    for label, value in rows
                ) + '<p>主题为候选发现。请在「研究复核」中确认原文、比较反例，再形成最终诠释。</p></aside>',
                unsafe_allow_html=True)


def theme_heading(index: int, theme: dict) -> None:
    st.markdown('<header class="tl-theme-heading">'
                f'<span class="tl-theme-number">{index:02d}</span><div>'
                f'<div class="tl-eyebrow">{escape(KIND_LABELS.get(theme.get("kind"), "待分类"))}</div>'
                f'<h3>{escape(str(theme.get("name") or "未命名主题"))}</h3></div></header>',
                unsafe_allow_html=True)


def narrative(text: str, label: str = "主题故事线", note: str = "模型草稿 · 待研究者核对") -> None:
    st.markdown(
        '<section class="tl-narrative">'
        f'<div class="tl-eyebrow">{escape(label)}</div>'
        f'<p>{escape(str(text))}</p><span class="tl-meta">{escape(note)}</span>'
        '</section>', unsafe_allow_html=True,
    )


def evidence_quote(code: dict, label: str = "原文引证") -> None:
    source = code.get("source") or {}
    details = [f"编码 [{code.get('code_id', '?')}]", source.get("source_id") or "来源未知"]
    if code.get("speaker"):
        details.append(str(code["speaker"]))
    if code.get("source_start") is not None:
        details.append(f"字符 {code['source_start']}–{code.get('source_end')}")
    st.markdown(
        '<figure class="tl-quote">'
        f'<div class="tl-eyebrow">{escape(label)}</div>'
        f'<blockquote>{escape(str(code.get("excerpt") or "没有有效摘录"))}</blockquote>'
        f'<figcaption>{escape(" · ".join(details))}</figcaption></figure>',
        unsafe_allow_html=True,
    )


def theme_badges(theme: dict, code_by_id: dict) -> None:
    counter_ids = set(theme.get("counterexample_code_ids") or [])
    support_ids = set(theme.get("code_ids") or []) - counter_ids
    supports = sum(is_matched(code_by_id.get(code_id, {})) for code_id in support_ids)
    counters = sum(is_matched(code_by_id.get(code_id, {})) for code_id in counter_ids)
    labels = [f"{supports} 条支持证据"]
    if counter_ids:
        labels.append(f"{counters} 条有效反证 / {len(counter_ids)} 条关联反例")
    st.markdown('<div class="tl-badges">' + ''.join(
        f'<span>{escape(label)}</span>' for label in labels
    ) + '</div>', unsafe_allow_html=True)


def theme_outline(themes: list, *, navigation: bool = False) -> None:
    if not themes:
        st.info("本次尚无候选主题。请在研究复核中检查原文匹配与待核查编码。")
        return
    rows = []
    for index, theme in enumerate(themes, 1):
        name = escape(str(theme.get("name") or "未命名主题"))
        kind = escape(KIND_LABELS.get(theme.get("kind"), "待分类"))
        title = f'<a href="#theme-{index}">{name}</a>' if navigation else f'<strong>{name}</strong>'
        description = '' if navigation else f'<p>{escape(str(theme.get("description") or ""))}</p>'
        rows.append(f'<li><span class="tl-index">{index:02d}</span><div>{title}{description}</div>'
                    f'<span class="tl-meta">{kind}</span></li>')
    tag = 'nav aria-label="主题目录"' if navigation else 'section aria-label="主题一览"'
    close = 'nav' if navigation else 'section'
    variant = "tl-outline-nav" if navigation else "tl-outline-summary"
    st.markdown(f'<{tag}><ol class="tl-outline {variant}">' + ''.join(rows) + f'</ol></{close}>',
                unsafe_allow_html=True)
