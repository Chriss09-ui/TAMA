import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.generation_agent import Chunk, Code, GenerationAgent, Theme, family_ids
from agents.refinement_agent import RefinementAgent
from agents.evaluation_agent import EvaluationAgent
from analysis_job import AnalysisCancelled, AnalysisJob
from threading import Barrier, Event
from evidence import is_matched, link_themes_to_codes, source_document, validate_documents, validate_evidence
from next_data_plan import build_next_data_plan
from reporting import report_lines
from research_records import ResearchWorkspace, SamplingTask, continuation_context, prepare_previous_result, render_research_records
from tama import TAMAFramework, ThreadlineFramework


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data, ensure_ascii=False)))])


def extracted(name, excerpt):
    return {"name": name, "description": name, "excerpt": excerpt,
            "statement_type": "participant_report", "verification_status": "reported_only"}


def theme(ids=(0,), counters=()):
    return {"name": "共享的条件", "description": "共享受条件限制", "code_ids": list(ids),
            "counterexample_code_ids": list(counters), "kind": "pattern"}


def finding(kind, name, excerpt):
    return {"kind": kind, "reason": "原文中的不同做法限制了候选解释", "code": extracted(name, excerpt)}


def evaluated():
    return {"theme_evaluations": [], "average_score": 4.5,
            "is_acceptable": True, "global_feedback": "候选主题符合评分要求"}


class EvidenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.agent = GenerationAgent(api_key="offline-test")
        self.agent.client = Mock()
        self.text = "受访者：我保存记录。"
        self.chunk = Chunk(chunk_id=0, text=self.text, start_word=0, end_word=12, end_char=len(self.text))

    def test_matching_real_statement_does_not_require_external_verification(self):
        code = self.agent.codes_from_data(self.chunk, [extracted("保存记录", "我保存记录")])[0]
        self.assertTrue(is_matched(code))
        self.assertEqual(code.verification_status, "reported_only")

    def test_missing_and_fabricated_quotes_are_archived_but_never_sent_as_evidence(self):
        codes = self.agent.codes_from_data(self.chunk, [extracted("虚构标签", "不存在"), extracted("缺失标签", None)])
        self.assertEqual([code.evidence_status for code in codes], ["mismatch", "missing"])
        self.agent.consolidate_codebook(codes)
        self.agent.map_codes(codes)
        self.assertEqual(self.agent.generate_themes(codes), [])
        self.agent.client.chat.completions.create.assert_not_called()
        self.assertEqual(len(codes), 2)

    def test_imported_flags_and_wrong_offsets_are_rechecked_against_original(self):
        code = Code(code_id=0, description="保存记录", source_chunks=[0], excerpt="我保存记录",
                    source={"source_id": "S"}, source_start=0, source_end=5, evidence_status="matched")
        document = source_document("S", self.text)
        validate_evidence([code], [document])
        self.assertEqual(code.evidence_status, "mismatch")
        code.source_start = 4
        code.source_end = 9
        validate_evidence([code], [document])
        self.assertEqual(code.evidence_status, "matched")
        validate_evidence([code], [])
        self.assertEqual(code.evidence_status, "unverified")

    def test_invalid_family_member_and_label_fallback_cannot_restore_support(self):
        good, bad = self.agent.codes_from_data(self.chunk, [extracted("保存记录", "我保存记录"), extracted("保存记录", "不存在")])
        good.merged_from = [1]
        bad.merged_into = 0
        by_id = {0: good, 1: bad}
        self.assertEqual(family_ids(0, by_id), [0])
        self.assertEqual(family_ids(1, by_id), [])
        linked = link_themes_to_codes([{"name": "模式", "code_ids": [1], "codes": ["保存记录"]}],
                                     [good.model_dump(), bad.model_dump()])[0]
        self.assertEqual(linked["code_ids"], [])
        self.assertEqual(linked["kind"], "evidence_gap")

    def test_source_body_tampering_is_rejected(self):
        document = source_document("S", self.text)
        document["text"] += "篡改"
        with self.assertRaises(ValueError):
            validate_documents([document])

    def test_legacy_record_without_body_stays_unverified(self):
        codes, _ = prepare_previous_result({"session_name": "legacy", "codes": [{"code_id": 0,
            "description": "保存记录", "excerpt": "我保存记录", "source_start": 4, "source_end": 9,
            "evidence_status": "matched"}]})
        self.assertEqual(codes[0]["evidence_status"], "unverified")

    def test_invalid_merged_member_is_detached_on_both_sides_and_can_be_imported_again(self):
        good, bad = self.agent.codes_from_data(self.chunk, [extracted("保存记录", "我保存记录"), extracted("保存记录", "不存在")])
        for code in (good, bad):
            code.source.source_id = "S"
        good.merged_from = [1]
        bad.merged_into = 0
        document = source_document("S", self.text)
        validate_evidence([good, bad], [document])
        self.assertEqual(good.merged_from, [])
        self.assertIsNone(bad.merged_into)
        restored, _ = prepare_previous_result({"codes": [good.model_dump(), bad.model_dump()],
                                               "source_documents": [document]})
        self.assertEqual([code["evidence_status"] for code in restored], ["matched", "mismatch"])

    def test_refinement_rejects_an_invalid_new_support_reference(self):
        good, bad = self.agent.codes_from_data(self.chunk, [extracted("保存记录", "我保存记录"), extracted("假证据", "不存在")])
        refiner = RefinementAgent(api_key="offline-test")
        refiner.client = Mock()
        refiner.client.chat.completions.create.return_value = response({"operations": [{
            "operation": "add", "target_themes": [], "rationale": "模拟建议",
            "new_theme": {"name": "虚构模式", "description": "不能通过", "code_ids": [1]}}], "summary": "检查"})
        result = refiner.run([], {"theme_evaluations": [], "global_feedback": ""},
                             [good.model_dump(), bad.model_dump()])
        self.assertEqual(result["refined_themes"][0]["kind"], "evidence_gap")
        prompt = refiner.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertNotIn("假证据", prompt)

    def test_evaluator_skips_unverified_evidence_even_with_a_candidate_theme(self):
        evaluator = EvaluationAgent(api_key="offline-test")
        evaluator.client = Mock()
        evaluator.decision_provider = Mock()
        result = evaluator.run([theme()], [{"code_id": 0, "description": "未经核对的标签"}])
        self.assertIsNone(result["average_score"])
        self.assertFalse(result["is_acceptable"])
        evaluator.client.chat.completions.create.assert_not_called()
        evaluator.decision_provider.ask.assert_not_called()


class FullTextWorkflowTests(unittest.TestCase):
    def make_framework(self, replies, **options):
        framework = ThreadlineFramework(api_key="offline-test", max_iterations=1, max_workers=1, **options)
        framework.generation_agent.client = Mock()
        framework.generation_agent.client.chat.completions.create.side_effect = [response(item) for item in replies]
        framework.evaluation_agent.run = Mock(return_value=evaluated())
        return framework

    def run_memory(self, framework, text):
        return framework.run_analysis(text, source_metadata={"source_id": "S"}, save_final=False)

    def test_default_review_recovers_a_missed_counterexample_and_updates_themes_once(self):
        text = "受访者：我经常共享记录。\n受访者：敏感记录从不共享。"
        framework = self.make_framework([
            {"codes": [extracted("共享记录", "我经常共享记录")]}, {"themes": [theme()]},
            {"findings": [finding("counterexample", "限制共享", "敏感记录从不共享")]},
            {"groups": []}, {}, {"themes": [theme((0, 1), (1,))]},
        ])
        result = self.run_memory(framework, text)
        self.assertTrue(result["configuration"]["corpus_review"])
        self.assertEqual(result["corpus_review"]["status"], "complete")
        self.assertEqual(result["corpus_review"]["added_code_ids"], [1])
        self.assertEqual(result["final_themes"][0]["counterexample_code_ids"], [1])
        self.assertTrue(result["accepted"])
        self.assertEqual(len(framework.generation_agent.client.chat.completions.create.call_args_list), 6)
        document = result["source_documents"][0]
        for code in result["codes"]:
            self.assertEqual(document["text"][code["source_start"]:code["source_end"]], code["excerpt"])
        self.assertEqual(validate_documents(result["source_documents"]), [source_document("S", text)])
        report = "\n".join(text for _, text in report_lines(result))
        self.assertIn("反例", report)
        self.assertNotIn(text, report)

    def test_review_runs_even_when_initial_extraction_is_empty(self):
        framework = self.make_framework([{"codes": []},
            {"findings": [finding("missed_support", "保存记录", "我保存记录")]}, {"themes": [theme()]}])
        result = self.run_memory(framework, "受访者：我保存记录。")
        self.assertEqual(result["corpus_review"]["added_code_ids"], [0])
        self.assertTrue(result["accepted"])

    def test_repeated_finding_reuses_existing_evidence_id(self):
        framework = self.make_framework([
            {"codes": [extracted("保存记录", "我保存记录")]}, {"themes": [theme()]},
            {"findings": [finding("context_limit", "保存记录", "我保存记录"),
                          finding("context_limit", "保存记录", "我保存记录")]},
        ])
        result = self.run_memory(framework, "受访者：我保存记录。")
        self.assertEqual(len(result["codes"]), 1)
        self.assertEqual(result["corpus_review"]["added_code_ids"], [])
        self.assertEqual([item["code_id"] for item in result["corpus_review"]["findings"]], [0, 0])

    def test_empty_and_invalid_extractions_return_unscored_savable_results(self):
        for records in ([], [extracted("不存在的经历", "虚构的摘录")]):
            with self.subTest(records=records), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "outputs"
                framework = self.make_framework([{"codes": records}, {"findings": []}], output_dir=str(output))
                result = framework.run_analysis("受访者：我保存记录。", save_final=True)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["stop_reason"], "no_valid_evidence")
                self.assertIsNone(result["metadata"]["final_average_score"])
                framework.evaluation_agent.run.assert_not_called()
                saved = json.loads((Path(result["output_dir"]) / "00_final_results.json").read_text())
                self.assertEqual(saved["source_documents"], result["source_documents"])
                self.assertIn("未评分", (Path(result["output_dir"]) / "00_summary.txt").read_text())
                if records:
                    self.assertIn("不存在的经历", (Path(result["output_dir"]) / "05_codebook.md").read_text())
                    self.assertNotIn("不存在的经历", (Path(result["output_dir"]) / "07_code_landscape.md").read_text())

    def test_partial_review_keeps_candidates_but_cannot_pass(self):
        text = "受访者：我保存记录。\n受访者：我不共享。"
        first, second = text.split("\n")
        framework = self.make_framework([])
        agent = framework.generation_agent
        agent.chunk_transcript = Mock(return_value=[
            Chunk(chunk_id=0, text=first, start_word=0, end_word=12, start_char=0, end_char=len(first)),
            Chunk(chunk_id=1, text=second, start_word=12, end_word=22, start_char=len(first)+1, end_char=len(text))])
        code = agent.codes_from_data(agent.chunk_transcript.return_value[0], [extracted("保存记录", "我保存记录")])[0]
        agent.generate_codes = Mock(return_value=[code])
        agent.generate_themes = Mock(return_value=[Theme(**{**theme(), "codes": ["保存记录"]})])
        agent.client.chat.completions.create.side_effect = [response({"findings": []}), ValueError("不得输出的响应正文")]
        result = self.run_memory(framework, text)
        self.assertEqual(result["corpus_review"]["status"], "partial")
        self.assertEqual(result["corpus_review"]["reviewed_chunks"], 1)
        self.assertEqual(result["corpus_review"]["failed_chunks"], [{"chunk_id": 1, "error_type": "ValueError"}])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["stop_reason"], "corpus_review_incomplete")
        self.assertNotIn("不得输出的响应正文", json.dumps(result, ensure_ascii=False))

    def test_malformed_review_is_explicit_failure(self):
        framework = self.make_framework([{"codes": [extracted("保存记录", "我保存记录")]},
                                         {"themes": [theme()]}, {"findings": "错误格式"}])
        result = self.run_memory(framework, "受访者：我保存记录。")
        self.assertEqual(result["corpus_review"]["status"], "failed")
        self.assertFalse(result["accepted"])
        self.assertEqual(len(result["final_themes"]), 1)

    def test_invalid_review_quote_stays_pending_and_never_rebuilds_themes(self):
        framework = self.make_framework([{"codes": []},
                                         {"findings": [finding("counterexample", "虚构反证", "不存在")]}])
        result = self.run_memory(framework, "受访者：我保存记录。")
        self.assertEqual(result["corpus_review"]["added_code_ids"], [])
        self.assertEqual(result["codes"][0]["evidence_status"], "mismatch")
        self.assertFalse(result["accepted"])
        framework.evaluation_agent.run.assert_not_called()

    def test_review_can_be_disabled_and_memory_only_run_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "outputs"
            framework = self.make_framework([{"codes": [extracted("保存记录", "我保存记录")]},
                                             {"themes": [theme()]}], corpus_review=False, output_dir=str(output))
            result = self.run_memory(framework, "受访者：我保存记录。")
            self.assertEqual(result["corpus_review"]["status"], "disabled")
            self.assertTrue(result["accepted"])
            self.assertFalse(output.exists())
            self.assertIs(TAMAFramework, ThreadlineFramework)

    def test_review_setting_requires_an_actual_boolean(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            ThreadlineFramework(api_key="offline-test", corpus_review="false")

    def test_cancel_at_review_checkpoint_is_not_swallowed(self):
        framework = self.make_framework([{"codes": [extracted("保存记录", "我保存记录")]}, {"themes": [theme()]}])
        def checkpoint(stage):
            if stage.startswith("全文回查"):
                raise AnalysisCancelled()
        with self.assertRaises(AnalysisCancelled):
            framework.run_analysis("受访者：我保存记录。", save_final=False, before_model_call=checkpoint)
        self.assertEqual(framework.generation_agent.client.chat.completions.create.call_count, 2)

    def test_two_independent_runs_do_not_reuse_codes_or_raw_text(self):
        framework = self.make_framework([{"codes": []}, {"findings": []}, {"codes": []}, {"findings": []}])
        first = self.run_memory(framework, "受访者：第一份记录。")
        second = self.run_memory(framework, "受访者：第二份记录。")
        self.assertNotEqual(first["session_name"], second["session_name"])
        self.assertNotIn("第一份记录", json.dumps(second, ensure_ascii=False))
        self.assertEqual(second["research_workspace"]["round_number"], 1)

    def test_cancel_during_last_review_request_stops_before_evaluation(self):
        framework = self.make_framework([])
        replies = iter([{"codes": [extracted("保存记录", "我保存记录")]}, {"themes": [theme()]}])
        def request(**kwargs):
            if "全文回查" not in kwargs["messages"][1]["content"] and "检查本次文稿" not in kwargs["messages"][1]["content"]:
                return response(next(replies))
            job.cancel()
            return response({"findings": []})
        framework.generation_agent.client.chat.completions.create.side_effect = request
        job = AnalysisJob(lambda checkpoint: framework.run_analysis("受访者：我保存记录。",
                          save_final=False, before_model_call=checkpoint))
        job.start()
        self.assertTrue(job.done.wait(3))
        self.assertTrue(job.snapshot().cancelled)
        self.assertIsNone(job.snapshot().result)
        framework.evaluation_agent.run.assert_not_called()

    def test_parallel_review_keeps_source_order_when_requests_finish_out_of_order(self):
        agent = GenerationAgent(api_key="offline-test", max_workers=3)
        agent.source_metadata = {"source_id": "S"}
        agent.client = Mock()
        barrier, last_finished = Barrier(3, timeout=3), Event()
        chunks = [Chunk(chunk_id=i, text=f"片段{i}", start_word=i, end_word=i+1,
                        start_char=i*3, end_char=(i+1)*3) for i in range(3)]
        def request(**kwargs):
            prompt = kwargs["messages"][1]["content"]
            index = next(i for i in range(3) if f"原文片段：\n片段{i}" in prompt)
            barrier.wait()
            if index == 0:
                self.assertTrue(last_finished.wait(3))
            if index == 2:
                last_finished.set()
            return response({"findings": [finding("missed_support", f"编码{index}", f"片段{index}")]})
        agent.client.chat.completions.create.side_effect = request
        codes = []
        result = agent.review_corpus(chunks, codes, [])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["added_code_ids"], [0, 1, 2])
        self.assertEqual([code.excerpt for code in codes], ["片段0", "片段1", "片段2"])


class MaterialScopeTests(unittest.TestCase):
    def test_historical_sampling_is_archived_but_not_in_prompts_or_report(self):
        task = SamplingTask(gap="历史缺口", target="历史采集目标", rationale="历史理由", researcher="R")
        workspace = ResearchWorkspace(sampling_tasks=[task])
        self.assertNotIn("历史采集目标", continuation_context(workspace))
        self.assertNotIn("历史采集目标", render_research_records(workspace))
        plan = build_next_data_plan([], [], [], workspace.model_dump())
        self.assertEqual(plan["tasks"][0]["target"], "历史采集目标")
        self.assertNotIn("历史采集目标", plan["markdown"])
        self.assertIn("当前材料复核", plan["markdown"])


if __name__ == "__main__":
    unittest.main()
