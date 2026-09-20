"""Local web interface for TAMA thematic analysis."""

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

import streamlit as st
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from openai import APIStatusError, OpenAI


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tama import TAMAFramework
from analysis_job import AnalysisJob


def api_model_name(provider: str, displayed_model: str) -> str:
    """Translate a branded DeepSeek version name to its API model ID."""
    model = displayed_model.strip()
    if provider == "DeepSeek" and model.casefold() == "deepseek-v4.1-flash":
        return "deepseek-flash"
    return model


def check_api_connection(api_key: str, model: str, base_url: str | None, provider: str) -> None:
    """Send one short request using the same endpoint and model as analysis."""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=20.0, max_retries=0)
    token_limit = {"max_completion_tokens": 256} if provider == "MiMo" else {"max_tokens": 128}
    response = client.chat.completions.create(
        model=api_model_name(provider, model),
        messages=[{"role": "user", "content": "请回复 OK。"}],
        **token_limit,
    )
    if not response.choices:
        raise ValueError("模型返回空结果")


@st.fragment(run_every="1s")
def render_active_job() -> None:
    job = st.session_state["analysis_job"]
    snapshot = job.snapshot()
    if snapshot.done:
        st.rerun()

    if snapshot.pause_requested:
        if st.button("继续分析", key="resume_analysis", use_container_width=True):
            job.resume()
            st.rerun()
    elif st.button("暂停分析", key="pause_analysis", use_container_width=True):
        job.pause()
        st.rerun()

    snapshot = job.snapshot()
    if snapshot.paused:
        st.warning(f"已暂停 · {snapshot.stage}")
    elif snapshot.pause_requested:
        st.info(f"等待当前请求完成后暂停 · {snapshot.stage}")
    else:
        st.info(f"正在分析 · {snapshot.stage}")
    st.caption("暂停会在下一次模型请求前生效；请保持页面和服务运行。")


def extract_docx_text(contents: bytes) -> str:
    """Extract body paragraphs and table rows in document order."""
    try:
        document = Document(BytesIO(contents))
    except (BadZipFile, PackageNotFoundError) as exc:
        raise ValueError("DOCX 文件无法读取，请确认文件未损坏且格式正确。") from exc

    lines = []
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                row_text = "\t".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    lines.append(row_text)
        elif block.text.strip():
            lines.append(block.text.strip())

    if not lines:
        raise ValueError("DOCX 文件中没有可读取的文字。图片或扫描件需要先进行文字识别。")
    return "\n".join(lines)


def read_transcript(source: str, pasted_text: str, uploaded_file) -> str:
    """Return pasted text or text extracted from a TXT or DOCX upload."""
    if source == "粘贴文本":
        return pasted_text.strip()
    if uploaded_file is None:
        return ""
    if Path(getattr(uploaded_file, "name", "")).suffix.lower() == ".docx":
        return extract_docx_text(uploaded_file.getvalue())
    return uploaded_file.getvalue().decode("utf-8-sig").strip()


def render_result(result: dict) -> None:
    """Display the latest analysis without exposing the API key."""
    st.divider()
    st.header("分析结果")
    st.caption("最近一次成功运行")

    status = "达到评估标准" if result["accepted"] else "已达到最大评估轮次"
    score = result["metadata"]["final_average_score"]
    themes = result["final_themes"]

    st.success(f"分析完成 · {status}")
    score_col, theme_col, round_col = st.columns(3)
    score_col.metric("平均分", f"{score:.2f} / 5")
    theme_col.metric("主题数", len(themes))
    round_col.metric("评估轮次", result["refinement_iterations"])

    st.download_button(
        "下载完整结果 JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{result['session_name']}.json",
        mime="application/json",
    )
    st.caption(f"本地结果目录：{ROOT / 'outputs' / result['session_name']}")

    if not themes:
        st.info("本次分析没有生成主题。请检查输入文本和模型返回内容。")
        return

    for index, theme in enumerate(themes, 1):
        with st.container(border=True):
            st.subheader(f"{index:02d} · {theme['name']}")
            st.write(theme["description"])
            codes = theme.get("codes", [])
            if codes:
                with st.expander(f"查看关联编码 · {len(codes)} 条"):
                    for code in codes:
                        st.write(f"• {code}")


def main() -> None:
    st.set_page_config(page_title="TAMA · 主题分析", page_icon="📝", layout="centered")
    job = st.session_state.get("analysis_job")
    running = job is not None and not job.snapshot().done

    with st.sidebar:
        st.header("模型设置")
        providers = {
            "DeepSeek": ("DEEPSEEK_API_KEY", "DeepSeek-V4.1-Flash"),
            "MiMo": ("MIMO_API_KEY", "mimo-v2.5-pro"),
            "OpenAI": ("OPENAI_API_KEY", "gpt-4o"),
        }
        provider = st.selectbox("服务商", list(providers), key="provider")
        env_name, default_model = providers[provider]
        api_key_input = st.text_input("API Key", type="password", key=f"api_key_{provider}")
        st.caption(f"留空时读取本机环境变量 {env_name}")

        model_key = f"model_{provider}"
        if provider == "DeepSeek" and "deepseek_display_migrated" not in st.session_state:
            if st.session_state.get(model_key) == "deepseek-flash":
                st.session_state[model_key] = "DeepSeek-V4.1-Flash"
            st.session_state["deepseek_display_migrated"] = True
        if model_key not in st.session_state:
            st.session_state[model_key] = default_model
        model = st.text_input("模型名称", key=model_key)
        if provider == "DeepSeek":
            st.caption("DeepSeek-V4.1-Flash 的 API 模型 ID 为 deepseek-flash；调用时自动转换。")
        base_url = None
        if provider == "MiMo":
            base_url = st.text_input(
                "接口地址",
                value=os.getenv("MIMO_BASE_URL") or "https://api.xiaomimimo.com/v1",
                help="使用 Token Plan 时，填写控制台提供的专属 OpenAI 兼容地址。",
                key="mimo_base_url",
            )
        elif provider == "DeepSeek":
            base_url = st.text_input(
                "接口地址",
                value=os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
                help="DeepSeek 官方 OpenAI 兼容接口地址；可按需填写自定义地址。",
                key="deepseek_base_url",
            )

        api_key = api_key_input.strip() or os.getenv(env_name, "").strip()
        endpoint = base_url.strip() if base_url is not None else None
        st.caption("测试会发送一次简短模型请求，可能产生少量费用。")
        if st.button("测试 API 连接", key="test_api_connection", disabled=running):
            if not api_key:
                st.error(f"请填写 API Key，或设置 {env_name} 环境变量。")
            elif not model.strip():
                st.error("请填写模型名称。")
            elif base_url is not None and not endpoint:
                st.error("请填写接口地址。")
            else:
                try:
                    with st.spinner("正在测试模型接口……"):
                        check_api_connection(api_key, model.strip(), endpoint, provider)
                except APIStatusError as exc:
                    st.error(f"连接失败 · HTTP {exc.status_code}，请检查密钥、地址和模型名称。")
                except Exception as exc:
                    st.error("连接失败，请检查网络、接口地址和模型名称。")
                    st.caption(f"错误类型：{type(exc).__name__}")
                else:
                    st.success("连接成功，密钥、地址和模型可用。")

        st.divider()
        st.header("运行设置")
        max_iterations = st.number_input("最多评估轮次", min_value=1, max_value=10, value=5)
        save_intermediate = st.checkbox(
            "保存阶段文件",
            value=False,
            help="包含文本片段、编码及逐轮评估，保存在本机 outputs 目录。",
        )

    st.caption("TAMA / THEMATIC ANALYSIS")
    st.title("访谈主题分析")
    st.write("输入访谈逐字稿，查看模型生成的编码、主题及评估结果。")

    st.header("访谈文本")
    source = st.radio("输入方式", ["粘贴文本", "上传文件"], horizontal=True)
    pasted_text = ""
    uploaded_file = None
    if source == "粘贴文本":
        pasted_text = st.text_area(
            "逐字稿内容",
            height=260,
            placeholder="在这里粘贴访谈逐字稿……",
            key="transcript_text",
        )
        if pasted_text.strip():
            st.caption(f"已输入 {len(pasted_text.strip())} 个字符")
    else:
        uploaded_file = st.file_uploader("选择 UTF-8 TXT 或 Word DOCX 文件", type=["txt", "docx"])

    st.caption("开始后，访谈文本会发送到所选模型服务进行分析。")
    if st.button("开始分析", type="primary", use_container_width=True, key="run_analysis", disabled=running):
        try:
            transcript = read_transcript(source, pasted_text, uploaded_file)
        except UnicodeDecodeError:
            st.error("文件无法按 UTF-8 读取，请将逐字稿另存为 UTF-8 文本后重试。")
        except ValueError as exc:
            st.error(str(exc))
        else:
            if not transcript:
                st.error("请先粘贴访谈文本或上传 TXT/DOCX 文件。")
            elif not api_key:
                st.error(f"请填写 API Key，或设置 {env_name} 环境变量。")
            elif not model.strip():
                st.error("请填写模型名称。")
            elif base_url is not None and not endpoint:
                st.error("请填写接口地址。")
            else:
                def run(checkpoint):
                    framework = TAMAFramework(
                        api_key=api_key,
                        model=api_model_name(provider, model),
                        base_url=endpoint,
                        max_iterations=int(max_iterations),
                        output_dir=str(ROOT / "outputs"),
                    )
                    return framework.run_analysis(
                        transcript=transcript,
                        save_intermediate=save_intermediate,
                        before_model_call=checkpoint,
                    )

                st.session_state.pop("analysis_result", None)
                job = AnalysisJob(run)
                st.session_state["analysis_job"] = job
                job.start()
                st.rerun()

    if job is not None and job.snapshot().done:
        snapshot = job.snapshot()
        if snapshot.result is not None:
            st.session_state["analysis_result"] = snapshot.result
        else:
            st.error("分析未完成。请检查密钥、接口地址、网络和模型返回内容。")
            st.caption(f"错误类型：{snapshot.error_type}")
            st.caption(f"失败阶段：{snapshot.stage}")
            if snapshot.error_location:
                st.caption(f"错误位置：{snapshot.error_location}")
            if snapshot.http_status:
                st.caption(f"HTTP 状态码：{snapshot.http_status}")
            if snapshot.error_type == "TypeError":
                st.info("程序调用发生类型错误。若刚更新过代码，请重启本地服务后重试。")
    elif running:
        render_active_job()

    if "analysis_result" in st.session_state:
        render_result(st.session_state["analysis_result"])
    else:
        st.divider()
        st.info("运行分析后，主题和关联编码会显示在这里。")


if __name__ == "__main__":
    main()
