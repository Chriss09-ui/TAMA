import sys
import unittest
from math import ceil
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.generation_agent import GenerationAgent
from chunking import plan_chunks


class ChunkingTests(unittest.TestCase):
    def test_chinese_without_spaces_is_split_by_adjustable_size(self):
        with patch("agents.generation_agent.OpenAI"):
            agent = GenerationAgent(api_key="test", chunk_size=7)

        chunks = agent.chunk_transcript("访谈者：你好。受访者：很好。")

        self.assertEqual([chunk.text for chunk in chunks], ["访谈者：你好。", "受访者：很好。"])
        self.assertEqual([(chunk.start_word, chunk.end_word) for chunk in chunks], [(0, 7), (7, 14)])

    def test_english_uses_words_and_keeps_original_text(self):
        with patch("agents.generation_agent.OpenAI"):
            agent = GenerationAgent(api_key="test", chunk_size=2)

        chunks = agent.chunk_transcript("one   two\nthree four five")

        self.assertEqual([chunk.text for chunk in chunks], ["one   two", "three four", "five"])

    def test_chunk_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "chunk_size must be positive"):
            GenerationAgent(api_key="test", chunk_size=0)

    def test_default_strategy_is_automatic_balanced(self):
        with patch("agents.generation_agent.OpenAI"):
            agent = GenerationAgent(api_key="test")

        chunks = agent.chunk_transcript("一段较短的访谈。")

        self.assertEqual(agent.chunk_strategy, "balanced")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(agent.last_chunk_plan.strategy, "balanced")

    def test_auto_profiles_trade_detail_for_fewer_chunks(self):
        transcript = "受访者描述了一段完整经历。" * 1000

        fine_plan, fine_chunks = plan_chunks(transcript, "fine")
        balanced_plan, balanced_chunks = plan_chunks(transcript, "balanced")
        economy_plan, economy_chunks = plan_chunks(transcript, "economy")

        self.assertGreaterEqual(len(fine_chunks), len(balanced_chunks))
        self.assertGreaterEqual(len(balanced_chunks), len(economy_chunks))
        self.assertLess(fine_plan.hard_limit, balanced_plan.hard_limit)
        self.assertLess(balanced_plan.hard_limit, economy_plan.hard_limit)
        for plan, chunks in (
            (fine_plan, fine_chunks),
            (balanced_plan, balanced_chunks),
            (economy_plan, economy_chunks),
        ):
            self.assertTrue(all(chunk.measured_size <= plan.hard_limit for chunk in chunks))

    def test_question_and_answer_stay_together_when_they_fit(self):
        transcript = (
            "访谈者：为什么？\n"
            "受访者：因为工作变化。\n"
            "访谈者：后来呢？\n"
            "受访者：后来逐渐适应了。"
        )

        plan, spans = plan_chunks(transcript, "manual", 20)
        texts = [transcript[span.start:span.end].strip() for span in spans]

        self.assertEqual(plan.num_chunks, 2)
        self.assertTrue(all(text.startswith("访谈者：") for text in texts))
        self.assertTrue(all("受访者：" in text for text in texts))

    def test_spans_cover_source_once_and_respect_hard_limit(self):
        transcript = (
            "访谈者：请描述最近的变化。\n受访者：工作安排发生了变化。\n\n"
            "访谈者：这带来了什么感受？\n受访者：开始有些焦虑，后来慢慢适应。\n\n"
            "访谈者：还有其他影响吗？\n受访者：通勤时间减少了。"
        )

        plan, spans = plan_chunks(transcript, "manual", 28)

        self.assertEqual(spans[0].start, 0)
        self.assertEqual(spans[-1].end, len(transcript))
        self.assertEqual(
            [(left.end, right.start) for left, right in zip(spans, spans[1:])],
            [(span.end, span.end) for span in spans[:-1]],
        )
        self.assertEqual("".join(transcript[span.start:span.end] for span in spans), transcript)
        self.assertTrue(all(span.measured_size <= plan.hard_limit for span in spans))
        if len(spans) > 1:
            self.assertGreaterEqual(spans[-1].measured_size, ceil(plan.target_size * 0.5))

    def test_manual_mode_requires_a_positive_size(self):
        with self.assertRaisesRegex(ValueError, "manual_chunk_size must be positive"):
            plan_chunks("访谈文本", "manual")


if __name__ == "__main__":
    unittest.main()
