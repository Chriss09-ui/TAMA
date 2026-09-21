import sys
import unittest
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.evaluation_agent import EvaluationAgent, EvaluationResult
from agents.generation_agent import Chunk, Code, GenerationAgent
from analysis_job import AnalysisJob


class ParallelRequestTests(unittest.TestCase):
    def test_code_extraction_runs_concurrently_and_keeps_chunk_order(self):
        with patch("agents.generation_agent.OpenAI"):
            agent = GenerationAgent(api_key="test", max_workers=3)
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

    def test_theme_evaluation_runs_concurrently_and_keeps_theme_order(self):
        with patch("agents.evaluation_agent.OpenAI"):
            agent = EvaluationAgent(api_key="test", max_workers=3)
        barrier = Barrier(3, timeout=3)
        last_finished = Event()
        themes = [{"name": f"主题 {i}", "description": "描述", "codes": []} for i in range(3)]

        def evaluate(theme, all_themes, original_codes):
            barrier.wait()
            if theme["name"] == "主题 0" and not last_finished.wait(3):
                raise TimeoutError("last theme did not finish first")
            if theme["name"] == "主题 2":
                last_finished.set()
            return EvaluationResult(
                theme_name=theme["name"], coverage_score=4, coverage_feedback="可以",
                actionability_score=4, actionability_feedback="可以",
                distinctiveness_score=4, distinctiveness_feedback="可以",
                relevance_score=4, relevance_feedback="可以", overall_score=4.0,
                needs_refinement=False, refinement_suggestions=[],
            )

        with patch.object(agent, "evaluate_theme", side_effect=evaluate):
            result = agent.evaluate_all_themes(themes, [])

        self.assertEqual([item.theme_name for item in result.theme_evaluations], [
            "主题 0", "主题 1", "主题 2",
        ])
        self.assertEqual(result.average_score, 4.0)

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
