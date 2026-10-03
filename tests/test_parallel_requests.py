import sys
import json
import unittest
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evidence_fixtures import matched_code
from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import Chunk, Code, GenerationAgent
from analysis_job import AnalysisJob
from decisions.base import DecisionAnswer, DecisionProvider, DecisionProviderError


class ParallelRequestTests(unittest.TestCase):
    def test_code_extraction_runs_concurrently_and_keeps_chunk_order(self):
        with patch("agents.generation_agent.OpenAI"):
            agent = GenerationAgent(api_key="test", max_workers=3)
        events = []
        agent.on_progress = events.append
        barrier = Barrier(3, timeout=3)
        last_finished = Event()
        chunks = [Chunk(chunk_id=i, text=str(i), start_word=i, end_word=i + 1) for i in range(3)]

        def extract(chunk):
            barrier.wait()
            if chunk.chunk_id == 0 and not last_finished.wait(3):
                raise TimeoutError("last chunk did not finish first")
            if chunk.chunk_id == 2:
                last_finished.set()
            return [Code(code_id=0, description=f"编码 {chunk.chunk_id}", source_chunks=[chunk.chunk_id])]

        with patch.object(agent, "generate_codes_from_chunk", side_effect=extract):
            codes = agent.generate_codes(chunks)

        self.assertEqual([code.code_id for code in codes], [0, 1, 2])
        self.assertEqual([code.description for code in codes], ["编码 0", "编码 1", "编码 2"])
        self.assertEqual(events[0].stage, "提取编码")
        self.assertEqual(events[0].total, 3)
        self.assertEqual({event.index for event in events if event.kind == "completed"}, {1, 2, 3})

    def test_theme_evaluation_runs_concurrently_and_keeps_theme_order(self):
        barrier = Barrier(3, timeout=3)
        finished = [Event() for _ in range(3)]
        completion_order = []
        requests = {}
        lock = Lock()
        themes = [{"name": f"主题 {i}", "description": "描述", "codes": [], "code_ids": [i]} for i in range(3)]
        codes = [matched_code(code_id=i, description=f"编码{i}").model_dump() for i in range(3)]

        class BlockingProvider(DecisionProvider):
            name = "blocking-stub"

            def ask(self, state, questions):
                index = state["theme"]["code_ids"][0]
                with lock:
                    requests[index] = (state, questions)
                barrier.wait()
                if index < 2 and not finished[index + 1].wait(3):
                    raise TimeoutError("next request did not finish first")
                answers = {key: DecisionAnswer(key=key, kind="score",
                    score=3.0 + index * 0.5 if key == "coverage" else 3.0, confidence=0.95)
                    for key in ("coverage", "actionability", "distinctiveness", "relevance")}
                answers["needs_refinement"] = DecisionAnswer(key="needs_refinement", kind="noul",
                    probability=0.1, confidence=0.9)
                with lock:
                    completion_order.append(index)
                finished[index].set()
                return answers

        with patch("agents.evaluation_agent.OpenAI"):
            agent = EvaluationAgent(api_key="test", max_workers=3, decision_provider=BlockingProvider())
        events = []
        agent.on_progress = events.append
        agent.client = Mock()
        result = agent.evaluate_all_themes(themes, codes)

        self.assertEqual(completion_order, [2, 1, 0])
        self.assertEqual([item.theme_name for item in result.theme_evaluations], [
            "主题 0", "主题 1", "主题 2",
        ])
        self.assertEqual([item.overall_score for item in result.theme_evaluations], [4.0, 4.125, 4.25])
        self.assertEqual([item.raw_scores["coverage"] for item in result.theme_evaluations], [3.0, 3.5, 4.0])
        self.assertEqual(result.average_score, 4.125)
        self.assertTrue(result.is_acceptable)
        self.assertEqual(set(requests), {0, 1, 2})
        for index, (state, questions) in requests.items():
            self.assertEqual(state["theme"]["name"], themes[index]["name"])
            self.assertEqual(state["original_codes"][index]["excerpt"], codes[index]["excerpt"])
            self.assertEqual(set(questions), {"coverage", "actionability", "distinctiveness", "relevance", "needs_refinement"})
        agent.client.chat.completions.create.assert_not_called()
        self.assertEqual(events[0].stage, "评估主题")
        self.assertEqual(events[0].total, 3)
        self.assertEqual({event.index for event in events if event.kind == "completed"}, {1, 2, 3})

    def test_initial_theme_batches_and_independent_merge_batches_run_concurrently(self):
        with patch("agents.generation_agent.OpenAI"):
            agent = GenerationAgent(api_key="test", max_workers=2)
        codes = [matched_code(code_id=i, description=f"编码{i}") for i in range(4)]
        initial_barrier = Barrier(2, timeout=3)
        merge_barrier = Barrier(2, timeout=3)
        later_initial_finished = Event()
        later_merge_finished = Event()
        final_candidates = []
        events = []
        agent.on_progress = events.append

        def request(prompt, stage):
            payload = json.loads(prompt)
            if stage.startswith("归纳主题"):
                index = payload[0]["code_id"] // 2
                initial_barrier.wait()
                if index == 0 and not later_initial_finished.wait(3):
                    raise TimeoutError("second theme batch did not finish")
                if index == 1:
                    later_initial_finished.set()
                return {"themes": [{"name": f"候选{item['code_id']}", "description": "候选描述",
                                     "code_ids": [item["code_id"]]} for item in payload]}
            if payload[0]["name"].startswith("候选"):
                index = int(payload[0]["name"][-1]) // 2
                merge_barrier.wait()
                if index == 0 and not later_merge_finished.wait(3):
                    raise TimeoutError("second merge batch did not finish")
                if index == 1:
                    later_merge_finished.set()
                return {"themes": [{"name": f"归并{index}", "description": "归并描述",
                                     "code_ids": [code_id for theme in payload for code_id in theme["code_ids"]]}]}
            final_candidates.extend(theme["name"] for theme in payload)
            return {"themes": payload}

        with patch("agents.generation_agent.THEME_BATCH_SIZE", 2), \
                patch.object(agent, "_theme_prompt", side_effect=json.dumps), \
                patch("agents.generation_agent.build_theme_consolidation_prompt", side_effect=lambda items, study: json.dumps(items)), \
                patch.object(agent, "_request_themes", side_effect=request):
            themes = agent.generate_themes(codes)

        self.assertEqual(final_candidates, ["归并0", "归并1"])
        self.assertEqual([theme.code_ids for theme in themes], [[0, 1], [2, 3]])
        phases = [event for event in events if event.kind == "phase"]
        self.assertEqual([(event.stage, event.total) for event in phases], [
            ("归纳主题", 2), ("合并候选主题", 2), ("合并候选主题", 1),
        ])
        self.assertEqual(len({event.batch_id for event in phases}), 3)

    def test_full_text_review_progress_flags_failed_chunks_and_keeps_valid_results(self):
        with patch("agents.generation_agent.OpenAI"):
            agent = GenerationAgent(api_key="test", max_workers=1)
        agent.source_metadata = {"source_id": "synthetic"}
        agent.client = Mock()
        agent.client.chat.completions.create.side_effect = [
            TimeoutError("synthetic timeout"),
            Mock(choices=[Mock(message=Mock(content='{"findings": []}'))]),
        ]
        events = []
        agent.on_progress = events.append
        chunks = [Chunk(chunk_id=i, text=str(i), start_word=i, end_word=i + 1) for i in range(2)]

        result = agent.review_corpus(chunks, [], [])

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["reviewed_chunks"], 1)
        completed = [event for event in events if event.kind == "completed"]
        self.assertEqual([(event.index, event.failed) for event in completed], [(1, True), (2, False)])

    def test_failed_theme_blocks_sibling_feedback_request_and_keeps_original_error(self):
        barrier = Barrier(2, timeout=3)
        failure_reported = Event()
        themes = [{"name": f"主题{i}", "description": "合成描述", "code_ids": [i]} for i in range(2)]
        codes = [matched_code(code_id=i, description=f"编码{i}").model_dump() for i in range(2)]

        class FailingProvider(DecisionProvider):
            name = "synthetic-failing"

            def ask(self, state, questions):
                barrier.wait()
                if state["theme"]["code_ids"] == [0]:
                    raise DecisionProviderError("synthetic failure", permanent=True)
                if not failure_reported.wait(3):
                    raise TimeoutError("failure was not reported")
                answers = {key: DecisionAnswer(key=key, kind="score", score=2.0, confidence=0.95)
                           for key in ("coverage", "actionability", "distinctiveness", "relevance")}
                answers["needs_refinement"] = DecisionAnswer(
                    key="needs_refinement", kind="noul", probability=0.9, confidence=0.95,
                )
                return answers

        with patch("agents.evaluation_agent.OpenAI"):
            agent = EvaluationAgent(api_key="test", max_workers=2, decision_provider=FailingProvider())
        agent.client = Mock()

        def progress(event):
            if event.kind == "completed" and event.failed:
                failure_reported.set()

        agent.on_progress = progress
        with self.assertRaisesRegex(DecisionProviderError, "synthetic failure"):
            agent.evaluate_all_themes(themes, codes)
        agent.client.chat.completions.create.assert_not_called()

    def test_pause_blocks_queued_parallel_requests(self):
        with patch("agents.generation_agent.OpenAI"):
            agent = GenerationAgent(api_key="test", max_workers=2)
        chunks = [Chunk(chunk_id=i, text=str(i), start_word=i, end_word=i + 1) for i in range(3)]
        first_two_started = Event()
        release_first_two = Event()
        third_started = Event()
        lock = Lock()
        started = []

        def extract(chunk):
            agent.before_model_call(f"提取编码 · 片段 {chunk.chunk_id + 1}")
            with lock:
                started.append(chunk.chunk_id)
                if len(started) == 2:
                    first_two_started.set()
            if chunk.chunk_id < 2:
                if not release_first_two.wait(3):
                    raise TimeoutError("first requests did not finish")
            else:
                third_started.set()
            return [Code(code_id=0, description=str(chunk.chunk_id), source_chunks=[chunk.chunk_id])]

        with patch.object(agent, "generate_codes_from_chunk", side_effect=extract):
            def run(checkpoint):
                agent.before_model_call = checkpoint
                return {"codes": agent.generate_codes(chunks)}

            job = AnalysisJob(run)
            job.start()
            try:
                self.assertTrue(first_two_started.wait(2))
                job.pause()
                release_first_two.set()
                self.assertFalse(third_started.wait(0.1))
                self.assertTrue(job.snapshot().pause_requested)
                job.resume()
                self.assertTrue(job.done.wait(3))
            finally:
                release_first_two.set()
                job.resume()

        self.assertTrue(third_started.is_set())
        self.assertIsNotNone(job.snapshot().result)


if __name__ == "__main__":
    unittest.main()
