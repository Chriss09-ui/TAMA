"""Presentation helpers for the white, typography-led research workspace."""

from html import escape
from pathlib import Path

import streamlit as st

from evidence import is_matched
from reporting import KIND_LABELS


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
