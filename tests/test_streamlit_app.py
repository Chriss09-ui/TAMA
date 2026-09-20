import os
import sys
import unittest
from io import BytesIO
from pathlib import Path
from threading import Event
from unittest.mock import ANY, patch

from docx import Document
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "streamlit_app.py"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from streamlit_app import api_model_name, check_api_connection, read_transcript


class StreamlitAppTests(unittest.TestCase):
    def test_uploaded_utf8_text_is_decoded(self):
        uploaded_file = BytesIO("\ufeff 访谈内容 \n".encode("utf-8"))
        self.assertEqual(read_transcript("上传文件", "", uploaded_file), "访谈内容")

    def test_uploaded_docx_reads_paragraphs_and_tables_in_order(self):
        document = Document()
        document.add_paragraph("访谈者：最近有什么变化？")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "受访者"
        table.cell(0, 1).text = "开始远程办公"
        document.add_paragraph("访谈者：谢谢。")

        uploaded_file = BytesIO()
        document.save(uploaded_file)
        uploaded_file.name = "访谈.DOCX"

        self.assertEqual(
            read_transcript("上传文件", "", uploaded_file),
            "访谈者：最近有什么变化？\n受访者\t开始远程办公\n访谈者：谢谢。",
        )

    def test_docx_without_body_text_reports_ocr_requirement(self):
        uploaded_file = BytesIO()
        Document().save(uploaded_file)
        uploaded_file.name = "扫描访谈.docx"

        with self.assertRaisesRegex(ValueError, "需要先进行文字识别"):
            read_transcript("上传文件", "", uploaded_file)

    def test_invalid_docx_reports_read_error(self):
        uploaded_file = BytesIO(b"not a DOCX file")
        uploaded_file.name = "访谈.docx"

        with self.assertRaisesRegex(ValueError, "DOCX 文件无法读取"):
            read_transcript("上传文件", "", uploaded_file)

    def test_empty_transcript_shows_validation_error(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.button(key="run_analysis").click().run()

        self.assertFalse(app.exception)
        self.assertIn("请先粘贴访谈文本", app.error[0].value)

    def test_analysis_displays_theme_without_network_request(self):
        result = {
            "session_name": "test-session",
            "accepted": True,
            "metadata": {"final_average_score": 4.25},
            "refinement_iterations": 1,
            "final_themes": [
                {"name": "治疗焦虑", "description": "受访者描述了治疗前的担忧。", "codes": ["术前担忧"]}
            ],
        }
        with patch.dict(os.environ, {}, clear=True), patch("tama.TAMAFramework") as framework:
            framework.return_value.run_analysis.return_value = result
            app = AppTest.from_file(str(APP_PATH)).run()
            app.selectbox(key="provider").set_value("MiMo").run()
            app.text_area(key="transcript_text").set_value("一段测试访谈文本").run()
            app.text_input(key="api_key_MiMo").set_value("test-key").run()
            app.button(key="run_analysis").click().run()
            self.assertTrue(app.session_state["analysis_job"].done.wait(2))
            app.run()

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["analysis_result"], result)
        framework.assert_called_once_with(
            api_key="test-key",
            model="mimo-v2.5-pro",
            base_url="https://api.xiaomimimo.com/v1",
            max_iterations=5,
            output_dir=str(ROOT / "outputs"),
        )
        framework.return_value.run_analysis.assert_called_once_with(
            transcript="一段测试访谈文本",
            save_intermediate=False,
            before_model_call=ANY,
        )

    def test_deepseek_provider_passes_model_and_endpoint(self):
        result = {
            "session_name": "deepseek-test-session",
            "accepted": True,
            "metadata": {"final_average_score": 4.0},
            "refinement_iterations": 1,
            "final_themes": [],
        }
        with patch.dict(os.environ, {}, clear=True), patch("tama.TAMAFramework") as framework:
            framework.return_value.run_analysis.return_value = result
            app = AppTest.from_file(str(APP_PATH)).run()
            self.assertEqual(app.selectbox(key="provider").value, "DeepSeek")
            self.assertEqual(app.text_input(key="model_DeepSeek").value, "DeepSeek-V4.1-Flash")
            app.text_area(key="transcript_text").set_value("一段测试访谈文本").run()
            app.text_input(key="api_key_DeepSeek").set_value("test-key").run()
            app.button(key="run_analysis").click().run()
            self.assertTrue(app.session_state["analysis_job"].done.wait(2))
            app.run()

        self.assertFalse(app.exception)
        framework.assert_called_once_with(
            api_key="test-key",
            model="deepseek-flash",
            base_url="https://api.deepseek.com",
            max_iterations=5,
            output_dir=str(ROOT / "outputs"),
        )
        framework.return_value.run_analysis.assert_called_once_with(
            transcript="一段测试访谈文本",
            save_intermediate=False,
            before_model_call=ANY,
        )

    def test_api_connection_uses_configured_endpoint_with_one_short_request(self):
        with patch("streamlit_app.OpenAI") as client:
            client.return_value.chat.completions.create.return_value.choices = [object()]
            check_api_connection("test-key", "DeepSeek-V4.1-Flash", "https://api.deepseek.com", "DeepSeek")

        client.assert_called_once_with(
            api_key="test-key", base_url="https://api.deepseek.com",
            timeout=20.0, max_retries=0,
        )
        client.return_value.chat.completions.create.assert_called_once_with(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "请回复 OK。"}],
            max_tokens=128,
        )

    def test_custom_deepseek_model_id_is_preserved(self):
        self.assertEqual(api_model_name("DeepSeek", "deepseek-v4-pro"), "deepseek-v4-pro")

    def test_existing_deepseek_default_is_relabelled_in_open_session(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(APP_PATH))
            app.session_state["provider"] = "DeepSeek"
            app.session_state["model_DeepSeek"] = "deepseek-flash"
            app.run()

        self.assertFalse(app.exception)
        self.assertEqual(app.text_input(key="model_DeepSeek").value, "DeepSeek-V4.1-Flash")

    def test_mimo_connection_uses_documented_completion_limit(self):
        with patch("streamlit_app.OpenAI") as client:
            client.return_value.chat.completions.create.return_value.choices = [object()]
            check_api_connection("test-key", "mimo-v2.5-pro", "https://api.xiaomimimo.com/v1", "MiMo")

        self.assertEqual(
            client.return_value.chat.completions.create.call_args.kwargs["max_completion_tokens"],
            256,
        )

    def test_pause_and_resume_buttons_control_running_analysis(self):
        first_started = Event()
        release_first = Event()
        second_started = Event()
        result = {
            "session_name": "paused-test", "accepted": True,
            "metadata": {"final_average_score": 4.0},
            "refinement_iterations": 1, "final_themes": [],
        }

        def run_analysis(*, before_model_call, **kwargs):
            before_model_call("提取编码")
            first_started.set()
            if not release_first.wait(10):
                raise TimeoutError("first model call timed out")
            before_model_call("归纳主题")
            second_started.set()
            return result

        with patch.dict(os.environ, {}, clear=True), patch("tama.TAMAFramework") as framework:
            framework.return_value.run_analysis.side_effect = run_analysis
            app = AppTest.from_file(str(APP_PATH)).run()
            app.selectbox(key="provider").set_value("MiMo").run()
            app.text_area(key="transcript_text").set_value("测试文本").run()
            app.text_input(key="api_key_MiMo").set_value("test-key").run()
            try:
                app.button(key="run_analysis").click().run(timeout=6)
                self.assertTrue(first_started.wait(2))
                app.button(key="pause_analysis").click().run()
                self.assertTrue(app.session_state["analysis_job"].snapshot().pause_requested)
                release_first.set()
                self.assertFalse(second_started.wait(0.05))
                app.button(key="resume_analysis").click().run()
                self.assertTrue(app.session_state["analysis_job"].done.wait(2))
                app.run()
            finally:
                release_first.set()
                if first_started.is_set():
                    app.session_state["analysis_job"].resume()

        self.assertFalse(app.exception)
        self.assertTrue(second_started.is_set())
        self.assertEqual(app.session_state["analysis_result"], result)

    def test_connection_button_uses_selected_provider_without_network(self):
        with patch.dict(os.environ, {}, clear=True), patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.return_value.choices = [object()]
            app = AppTest.from_file(str(APP_PATH)).run()
            app.text_input(key="api_key_DeepSeek").set_value("test-key").run()
            app.button(key="test_api_connection").click().run()

        self.assertFalse(app.exception)
        self.assertIn("连接成功", app.success[0].value)
        client.assert_called_once_with(
            api_key="test-key", base_url="https://api.deepseek.com",
            timeout=20.0, max_retries=0,
        )
        self.assertEqual(
            client.return_value.chat.completions.create.call_args.kwargs["model"],
            "deepseek-flash",
        )

    def test_connection_error_does_not_show_raw_exception(self):
        with patch.dict(os.environ, {}, clear=True), patch("openai.OpenAI") as client:
            client.return_value.chat.completions.create.side_effect = RuntimeError("test-key")
            app = AppTest.from_file(str(APP_PATH)).run()
            app.selectbox(key="provider").set_value("MiMo").run()
            app.text_input(key="api_key_MiMo").set_value("test-key").run()
            app.button(key="test_api_connection").click().run()

        self.assertFalse(app.exception)
        self.assertIn("连接失败", app.error[0].value)
        self.assertNotIn("test-key", app.error[0].value)

    def test_analysis_type_error_shows_safe_diagnostics(self):
        with patch.dict(os.environ, {}, clear=True), patch("tama.TAMAFramework") as framework:
            framework.return_value.run_analysis.side_effect = TypeError("secret-api-key")
            app = AppTest.from_file(str(APP_PATH)).run()
            app.text_area(key="transcript_text").set_value("测试文本").run()
            app.text_input(key="api_key_DeepSeek").set_value("test-key").run()
            app.button(key="run_analysis").click().run()
            self.assertTrue(app.session_state["analysis_job"].done.wait(2))
            app.run()

        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("错误类型：TypeError", captions)
        self.assertIn("错误位置：", captions)
        self.assertNotIn("secret-api-key", captions)


if __name__ == "__main__":
    unittest.main()
