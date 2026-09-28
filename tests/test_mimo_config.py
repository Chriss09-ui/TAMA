import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from example_usage import get_model_config
from tama import TAMAFramework
from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import GenerationAgent
from agents.refinement_agent import RefinementAgent
from research_profile import ResearchProfile


class MiMoConfigTests(unittest.TestCase):
    def test_example_selects_mimo_with_default_endpoint(self):
        with patch.dict(os.environ, {"MIMO_API_KEY": "mimo-test"}, clear=True):
            self.assertEqual(
                get_model_config(),
                ("mimo-test", "mimo-v2.5-pro", "https://api.xiaomimimo.com/v1")
            )

    def test_example_uses_token_plan_endpoint(self):
        with patch.dict(
            os.environ,
            {"MIMO_API_KEY": "mimo-test", "MIMO_BASE_URL": "https://plan.example/v1"},
            clear=True
        ):
            self.assertEqual(get_model_config()[2], "https://plan.example/v1")

    def test_example_keeps_openai_default(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test"}, clear=True):
            self.assertEqual(get_model_config(), ("openai-test", "gpt-4o", None))

    def test_framework_passes_endpoint_to_all_agents(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with patch("tama.GenerationAgent") as generation, \
                    patch("tama.EvaluationAgent") as evaluation, \
                    patch("tama.RefinementAgent") as refinement:
                TAMAFramework(
                    api_key="mimo-test",
                    model="mimo-v2.5-pro",
                    output_dir=output_dir,
                    base_url="https://api.xiaomimimo.com/v1",
                    chunk_size=600,
                    max_workers=2,
                )

        generation.assert_called_once_with(
            api_key="mimo-test", model="mimo-v2.5-pro",
            base_url="https://api.xiaomimimo.com/v1", chunk_size=600,
            chunk_strategy="manual",
            max_workers=2,
            study=ResearchProfile(),
        )
        evaluation.assert_called_once_with(
            api_key="mimo-test", model="mimo-v2.5-pro", expert_criteria=None,
            base_url="https://api.xiaomimimo.com/v1", max_workers=2,
            decision_provider=None, confidence_threshold=0.7,
            study=ResearchProfile(),
        )
        refinement.assert_called_once_with(
            api_key="mimo-test", model="mimo-v2.5-pro",
            base_url="https://api.xiaomimimo.com/v1", study=ResearchProfile(),
        )

    def test_each_agent_constructs_client_with_endpoint(self):
        for module, agent in (
            ("agents.generation_agent", GenerationAgent),
            ("agents.refinement_agent", RefinementAgent),
        ):
            with self.subTest(agent=agent.__name__):
                with patch(f"{module}.OpenAI") as client:
                    agent(api_key="mimo-test", model="mimo-v2.5-pro", base_url="https://api.xiaomimimo.com/v1")
                client.assert_called_once_with(
                    api_key="mimo-test", base_url="https://api.xiaomimimo.com/v1"
                )

        # The evaluation agent also builds its default JSON-mode decision
        # client from the same credentials; patch it out to isolate the agent
        # client construction.
        with patch("agents.evaluation_agent.OpenAI") as client, \
                patch("agents.evaluation_agent.LLMDecisionClient"):
            EvaluationAgent(
                api_key="mimo-test", model="mimo-v2.5-pro",
                base_url="https://api.xiaomimimo.com/v1",
            )
        client.assert_called_once_with(
            api_key="mimo-test", base_url="https://api.xiaomimimo.com/v1"
        )


if __name__ == "__main__":
    unittest.main()
