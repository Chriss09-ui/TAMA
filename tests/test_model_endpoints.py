"""Connection settings at the framework-to-agent and agent-to-client boundaries."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tama import ThreadlineFramework
from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import GenerationAgent
from agents.refinement_agent import RefinementAgent
from research_profile import ResearchProfile


class ModelEndpointTests(unittest.TestCase):
    def test_framework_passes_endpoint_to_all_agents(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with patch("tama.GenerationAgent") as generation, \
                    patch("tama.EvaluationAgent") as evaluation, \
                    patch("tama.RefinementAgent") as refinement:
                ThreadlineFramework(
                    api_key="mimo-test",
                    model="mimo-v2.6-pro",
                    output_dir=output_dir,
                    base_url="https://api.xiaomimimo.com/v1",
                    chunk_size=600,
                    max_workers=2,
                )

        generation.assert_called_once_with(
            api_key="mimo-test", model="mimo-v2.6-pro",
            base_url="https://api.xiaomimimo.com/v1", chunk_size=600,
            chunk_strategy="manual",
            max_workers=2,
            study=ResearchProfile(),
        )
        evaluation.assert_called_once_with(
            api_key="mimo-test", model="mimo-v2.6-pro", expert_criteria=None,
            base_url="https://api.xiaomimimo.com/v1", max_workers=2,
            decision_provider=None, confidence_threshold=0.7,
            study=ResearchProfile(),
        )
        refinement.assert_called_once_with(
            api_key="mimo-test", model="mimo-v2.6-pro",
            base_url="https://api.xiaomimimo.com/v1", study=ResearchProfile(),
        )

    def test_each_agent_constructs_client_with_endpoint(self):
        for module, agent in (
            ("agents.generation_agent", GenerationAgent),
            ("agents.refinement_agent", RefinementAgent),
        ):
            with self.subTest(agent=agent.__name__):
                with patch(f"{module}.OpenAI") as client:
                    agent(api_key="mimo-test", model="mimo-v2.6-pro", base_url="https://api.xiaomimimo.com/v1")
                client.assert_called_once_with(
                    api_key="mimo-test", base_url="https://api.xiaomimimo.com/v1"
                )

        # The evaluation agent also builds its default JSON-mode decision
        # client from the same credentials; patch it out to isolate the agent
        # client construction.
        with patch("agents.evaluation_agent.OpenAI") as client, \
                patch("agents.evaluation_agent.LLMDecisionClient"):
            EvaluationAgent(
                api_key="mimo-test", model="mimo-v2.6-pro",
                base_url="https://api.xiaomimimo.com/v1",
            )
        client.assert_called_once_with(
            api_key="mimo-test", base_url="https://api.xiaomimimo.com/v1"
        )


if __name__ == "__main__":
    unittest.main()
