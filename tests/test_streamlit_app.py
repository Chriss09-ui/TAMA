import json
import os
import sys
import unittest
from io import BytesIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import ANY, patch
from xml.etree import ElementTree
from zipfile import ZipFile

from docx import Document
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "streamlit_app.py"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from streamlit_app import KEYRING_SERVICE, api_model_name, build_result_docx, check_api_connection, read_transcript


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        keyring_patch = patch("keyring.get_password", return_value=None)
        self.keyring_get = keyring_patch.start()
        self.addCleanup(keyring_patch.stop)

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

    def test_docx_with_missing_auxiliary_part_recovers_body_text(self):
        document = Document()
        document.add_paragraph("访谈者：最近有什么变化？")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "受访者"
        table.cell(0, 1).text = "开始远程办公"
        original = BytesIO()
        document.save(original)

        damaged = BytesIO()
        with ZipFile(BytesIO(original.getvalue())) as source, ZipFile(damaged, "w") as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "_rels/.rels":
                    relationships = ElementTree.fromstring(data)
                    auxiliary = next(
                        rel for rel in relationships
                        if rel.attrib.get("Target", "").endswith("core.xml")
                    )
                    auxiliary.set("Target", "NULL")
                    data = ElementTree.tostring(relationships, encoding="utf-8", xml_declaration=True)
                target.writestr(item, data)

        with self.assertRaises(KeyError):
            Document(BytesIO(damaged.getvalue()))
        damaged.name = "访谈.docx"
        self.assertEqual(
            read_transcript("上传文件", "", damaged),
            "访谈者：最近有什么变化？\n受访者\t开始远程办公",
        )

    def test_docx_without_body_part_reports_clear_error(self):
        original = BytesIO()
        Document().save(original)
        damaged = BytesIO()
        with ZipFile(BytesIO(original.getvalue())) as source, ZipFile(damaged, "w") as target:
            for item in source.infolist():
                if item.filename != "word/document.xml":
                    target.writestr(item, source.read(item.filename))
        damaged.name = "访谈.docx"

        with self.assertRaisesRegex(ValueError, "DOCX 文件结构异常，无法读取正文"):
            read_transcript("上传文件", "", damaged)

    def test_saved_word_report_contains_themes_and_codes(self):
        result = {
            "session_name": "report-test", "accepted": True,
            "metadata": {"final_average_score": 4.25},
            "refinement_iterations": 1,
            "final_evaluation": {
                "global_feedback": "主题整体达标，但有一项置信度偏低。",
                "flagged_themes": ["工作方式变化"],
            },
            "final_themes": [{
                "name": "工作方式变化", "description": "受访者描述了远程办公。",
                "codes": ["通勤时间减少", "线上沟通增加"],
            }],
        }

        document = Document(BytesIO(build_result_docx(result)))
        paragraphs = "\n".join(paragraph.text for paragraph in document.paragraphs)
        for expected in ("访谈主题分析报告", "工作方式变化", "受访者描述了远程办公。",
                         "通勤时间减少", "线上沟通增加", "建议人工复核"):
            self.assertIn(expected, paragraphs)

    def test_result_page_offers_word_and_json_saves(self):
        result = {
            "session_name": "report-test", "accepted": True,
            "metadata": {"final_average_score": 4.25},
            "refinement_iterations": 1, "final_themes": [],
            "final_evaluation": {
                "global_feedback": "主题已达标，但建议复核。",
                "flagged_themes": ["主题甲"],
            },
        }
        with patch.dict(os.environ, {}, clear=True), patch("streamlit.download_button") as download:
            app = AppTest.from_file(str(APP_PATH))
            app.session_state["analysis_result"] = result
            app.run()

        self.assertFalse(app.exception)
        self.assertEqual([call.args[0] for call in download.call_args_list], [
            "保存 Word 报告（DOCX）", "保存完整数据（JSON）",
        ])
        self.assertEqual(download.call_args_list[0].kwargs["file_name"], "report-test.docx")
        self.assertEqual(json.loads(download.call_args_list[1].kwargs["data"]), result)
        self.assertIn("主题甲", app.warning[0].value)

    def test_empty_transcript_shows_validation_error(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.button(key="run_analysis").click().run()

        self.assertFalse(app.exception)
        self.assertIn("请先粘贴访谈文本", app.error[0].value)

    def test_page_shows_methodology_references_at_the_bottom(self):
        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(APP_PATH)).run()

        self.assertFalse(app.exception)
        headings = [item.value for item in app.header]
        self.assertLess(headings.index("1. 输入访谈材料"), headings.index("2. 运行分析"))
        self.assertLess(headings.index("2. 运行分析"), headings.index("3. 分析结果"))
        self.assertLess(headings.index("3. 分析结果"), headings.index("方法依据与参考文献"))
        references = next(
            item for item in app.expander if item.label == "查看切块策略说明和文献"
        ).markdown[0].value
        self.assertIn("TAMA 原始方法", references)
        self.assertIn("Lost in the Middle", references)
        self.assertIn("Token 用量计算", references)

        app.text_area(key="transcript_text").set_value("访谈者：工作有什么变化？\n受访者：开始远程办公。 ").run()
        self.assertIn("切块预估", [item.value for item in app.subheader])
        self.assertTrue(any("预计" in item.value and "片段" in item.value for item in app.markdown))

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
            self.assertEqual(app.selectbox(key="chunk_strategy").value, "自动 · 均衡")
            app.selectbox(key="chunk_strategy").set_value("手动设置").run()
            self.assertEqual(app.number_input(key="chunk_size").value, 4000)
            app.number_input(key="chunk_size").set_value(600).run()
            self.assertEqual(app.number_input(key="max_workers").value, 4)
            app.number_input(key="max_workers").set_value(2).run()
            app.button(key="run_analysis").click().run()
            self.assertTrue(app.session_state["analysis_job"].done.wait(2))
            app.run()

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["analysis_result"], result)
        framework.assert_called_once_with(
            api_key="test-key",
            model="mimo-v2.5-pro",
            base_url="https://api.xiaomimimo.com/v1",
            chunk_size=600,
            chunk_strategy="manual",
            max_workers=2,
            max_iterations=5,
            decision_provider=None,
            confidence_threshold=0.7,
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
            chunk_size=None,
            chunk_strategy="balanced",
            max_workers=4,
            max_iterations=5,
            decision_provider=None,
            confidence_threshold=0.7,
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
                self.assertTrue(app.selectbox(key="provider").disabled)
                self.assertTrue(app.radio[0].disabled)
                self.assertTrue(app.text_area(key="transcript_text").disabled)
                self.assertTrue(app.text_input(key="model_MiMo").disabled)
                self.assertTrue(app.number_input(key="max_workers").disabled)
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

    def test_saved_key_is_used_after_new_browser_session(self):
        vault = {}
        self.keyring_get.side_effect = lambda service, provider: vault.get((service, provider))
        with patch("keyring.set_password", side_effect=lambda service, provider, key: vault.__setitem__((service, provider), key)), \
                patch.dict(os.environ, {}, clear=True), patch("tama.TAMAFramework") as framework:
            framework.return_value.run_analysis.return_value = {
                "session_name": "saved-key-test", "accepted": True,
                "metadata": {"final_average_score": 4.0},
                "refinement_iterations": 1, "final_themes": [],
            }
            app = AppTest.from_file(str(APP_PATH)).run()
            app.text_input(key="api_key_DeepSeek").set_value(" saved-test-key ").run()
            app.button(key="save_api_key").click().run()
            self.assertEqual(vault[(KEYRING_SERVICE, "DeepSeek")], "saved-test-key")
            self.assertEqual(app.text_input(key="api_key_DeepSeek").value, "")

            new_session = AppTest.from_file(str(APP_PATH)).run()
            new_session.text_area(key="transcript_text").set_value("测试访谈").run()
            new_session.button(key="run_analysis").click().run()
            self.assertTrue(new_session.session_state["analysis_job"].done.wait(2))
            new_session.run()

        self.assertFalse(new_session.exception)
        framework.assert_called_once_with(
            api_key="saved-test-key", model="deepseek-flash",
            base_url="https://api.deepseek.com", chunk_size=None,
            chunk_strategy="balanced",
            max_workers=4, max_iterations=5,
            decision_provider=None, confidence_threshold=0.7,
            output_dir=str(ROOT / "outputs"),
        )

    def test_delete_saved_key_removes_it_from_next_session(self):
        vault = {(KEYRING_SERVICE, "MiMo"): "saved-test-key"}
        self.keyring_get.side_effect = lambda service, provider: vault.get((service, provider))
        with patch("keyring.delete_password", side_effect=lambda service, provider: vault.pop((service, provider))), \
                patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file(str(APP_PATH)).run()
            app.selectbox(key="provider").set_value("MiMo").run()
            app.button(key="delete_saved_api_key").click().run()
            self.assertNotIn((KEYRING_SERVICE, "MiMo"), vault)

            new_session = AppTest.from_file(str(APP_PATH)).run()
            new_session.selectbox(key="provider").set_value("MiMo").run()
            new_session.text_area(key="transcript_text").set_value("测试访谈").run()
            new_session.button(key="run_analysis").click().run()

        self.assertFalse(new_session.exception)
        self.assertIn("请填写 API Key", new_session.error[0].value)

    def test_keyring_failure_allows_manual_key_without_exposing_error(self):
        self.keyring_get.side_effect = RuntimeError("secret-from-keyring")
        with patch.dict(os.environ, {}, clear=True), patch("keyring.set_password", side_effect=RuntimeError("secret-from-keyring")):
            app = AppTest.from_file(str(APP_PATH)).run()
            self.assertIn("系统凭据库暂不可用", app.warning[0].value)
            self.assertNotIn("secret-from-keyring", app.warning[0].value)
            app.text_input(key="api_key_DeepSeek").set_value("test-key").run()
            app.button(key="save_api_key").click().run()

        self.assertFalse(app.exception)
        self.assertIn("保存失败", app.error[0].value)
        self.assertNotIn("secret-from-keyring", app.error[0].value)

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

    def test_jev_mode_without_key_blocks_analysis(self):
        with patch.dict(os.environ, {}, clear=True), patch("tama.TAMAFramework") as framework:
            app = AppTest.from_file(str(APP_PATH)).run()
            app.selectbox(key="decision_mode").set_value("Jev（实验性）").run()
            app.text_area(key="transcript_text").set_value("测试访谈").run()
            app.text_input(key="api_key_DeepSeek").set_value("test-key").run()
            app.button(key="run_analysis").click().run()

        self.assertFalse(app.exception)
        self.assertIn("Jev API Key", app.error[0].value)
        framework.assert_not_called()

    def test_jev_mode_builds_jev_decision_client(self):
        result = {
            "session_name": "jev-test", "accepted": True,
            "metadata": {"final_average_score": 4.0},
            "refinement_iterations": 1, "final_themes": [],
        }
        with patch.dict(os.environ, {}, clear=True), \
                patch("tama.TAMAFramework") as framework, \
                patch("decisions.jev_client.JevDecisionClient") as jev_client_cls:
            framework.return_value.run_analysis.return_value = result
            app = AppTest.from_file(str(APP_PATH)).run()
            app.selectbox(key="decision_mode").set_value("Jev（实验性）").run()
            app.text_input(key="api_key_Jev").set_value("jev-test").run()
            app.text_area(key="transcript_text").set_value("测试访谈").run()
            app.text_input(key="api_key_DeepSeek").set_value("test-key").run()
            app.button(key="run_analysis").click().run()
            self.assertTrue(app.session_state["analysis_job"].done.wait(2))
            app.run()

        self.assertFalse(app.exception)
        jev_client_cls.assert_called_once_with(api_key="jev-test", base_url=None, model=None)
        framework.assert_called_once_with(
            api_key="test-key", model="deepseek-flash",
            base_url="https://api.deepseek.com", chunk_size=None,
            chunk_strategy="balanced",
            max_workers=4, max_iterations=5,
            decision_provider=jev_client_cls.return_value,
            confidence_threshold=0.7,
            output_dir=str(ROOT / "outputs"),
        )

    def test_jev_connection_test_sends_noul_request(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("openai.OpenAI") as main_client, \
                patch("decisions.jev_client.JevDecisionClient") as jev_client_cls:
            main_client.return_value.chat.completions.create.return_value.choices = [object()]
            jev_client_cls.return_value.ask.return_value = {
                "check": SimpleNamespace(probability=0.99),
            }
            app = AppTest.from_file(str(APP_PATH)).run()
            app.selectbox(key="decision_mode").set_value("Jev（实验性）").run()
            app.text_input(key="api_key_Jev").set_value("jev-test").run()
            app.text_input(key="api_key_DeepSeek").set_value("test-key").run()
            app.button(key="test_api_connection").click().run()

        self.assertFalse(app.exception)
        self.assertIn("连接成功", app.success[0].value)
        jev_client_cls.assert_called_once_with(
            api_key="jev-test", base_url=None, timeout=20.0, max_retries=0,
        )
        state, questions = jev_client_cls.return_value.ask.call_args.args
        self.assertEqual(state, "连接测试")
        self.assertEqual(questions["check"].kind, "noul")


if __name__ == "__main__":
    unittest.main()
