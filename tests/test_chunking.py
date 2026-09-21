import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.generation_agent import GenerationAgent


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


if __name__ == "__main__":
    unittest.main()
