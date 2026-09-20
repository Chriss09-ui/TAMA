"""
TAMA: A Human-AI Collaborative Thematic Analysis Framework Using Multi-Agent LLMs
Main orchestrator coordinating Generation, Evaluation, and Refinement agents.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
import json

from agents.generation_agent import GenerationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.refinement_agent import RefinementAgent


class TAMAFramework:
    """
    TAMA Framework orchestrator that coordinates multi-agent workflow:
    1. Generation Agent: Chunks -> Codes -> Themes
    2. Evaluation Agent: Evaluates themes against criteria
    3. Refinement Agent: Refines themes based on feedback
    4. Iterates until affirmative answer (acceptable themes) or max iterations
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_iterations: int = 5,
        acceptance_threshold: float = 4.0,
        output_dir: str = "outputs",
        expert_criteria: Optional[Dict[str, str]] = None
    ):
        """
        Initialize TAMA Framework.

        Args:
            api_key: OpenAI API key
            model: Model to use for all agents (default: gpt-4o)
            max_iterations: Maximum refinement iterations (default: 5)
            acceptance_threshold: Minimum score to accept themes (default: 4.0)
            output_dir: Directory to save outputs (default: outputs)
            expert_criteria: Optional custom evaluation criteria from cardiac expert
        """
        self.api_key = api_key
        self.model = model
        self.max_iterations = max_iterations
        self.acceptance_threshold = acceptance_threshold
        self.output_dir = output_dir
        self.expert_criteria = expert_criteria

        # Initialize agents
        self.generation_agent = GenerationAgent(api_key=api_key, model=model)
        self.evaluation_agent = EvaluationAgent(
            api_key=api_key,
            model=model,
            expert_criteria=expert_criteria
        )
        self.refinement_agent = RefinementAgent(api_key=api_key, model=model)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def run_analysis(
        self,
        transcript: str,
        session_name: str = None,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete TAMA analysis with iterative refinement.

        Args:
            transcript: Interview transcript text
            session_name: Optional name for this analysis session
            save_intermediate: Whether to save intermediate results

        Returns:
            Dictionary containing final themes and analysis metadata
        """
        if session_name is None:
            session_name = f"tama_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session_dir = os.path.join(self.output_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)

        print("=" * 80)
        print(f"TAMA FRAMEWORK - Session: {session_name}")
        print("=" * 80)

        # Phase 1: Generation
        print("\n" + "=" * 80)
        print("PHASE 1: GENERATION AGENT")
        print("=" * 80)

        generation_path = os.path.join(session_dir, "01_generation.json") if save_intermediate else None
        generation_result = self.generation_agent.run(transcript, save_path=generation_path)

        themes = generation_result["themes"]
        codes = generation_result["codes"]

        # Phase 2: Iterative Evaluation and Refinement
        iteration = 0
        is_acceptable = False
        refinement_history = []

        while not is_acceptable and iteration < self.max_iterations:
            iteration += 1

            print("\n" + "=" * 80)
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print("=" * 80)

            # Evaluation
            print("\n" + "-" * 80)
            print("PHASE 2: EVALUATION AGENT")
            print("-" * 80)

            eval_path = os.path.join(
                session_dir,
                f"02_evaluation_iter{iteration}.json"
            ) if save_intermediate else None

            evaluation_result = self.evaluation_agent.run(
                themes=themes,
                codes=codes,
                save_path=eval_path
            )

            print(f"\nEvaluation Results:")
            print(f"  Average Score: {evaluation_result['average_score']:.2f}/5.0")
            print(f"  Acceptable: {evaluation_result['is_acceptable']}")
            print(f"  Feedback: {evaluation_result['global_feedback']}")

            is_acceptable = evaluation_result["is_acceptable"]

            # If acceptable or max iterations reached, stop
            if is_acceptable:
                print("\n✓ Themes are acceptable! Analysis complete.")
                break

            if iteration >= self.max_iterations:
                print(f"\n⚠ Maximum iterations ({self.max_iterations}) reached.")
                print("  Proceeding with current themes.")
                break

            # Refinement
            print("\n" + "-" * 80)
            print("PHASE 3: REFINEMENT AGENT")
            print("-" * 80)

            refinement_path = os.path.join(
                session_dir,
                f"03_refinement_iter{iteration}.json"
            ) if save_intermediate else None

            refinement_result = self.refinement_agent.run(
                themes=themes,
                evaluation_results=evaluation_result,
                codes=codes,
                save_path=refinement_path
            )

            # Update themes for next iteration
            themes = refinement_result["refined_themes"]

            # Track refinement history
            refinement_history.append({
                "iteration": iteration,
                "evaluation": evaluation_result,
                "refinement": refinement_result
            })

            print(f"\nPreparing for iteration {iteration + 1}...")

        # Phase 3: Finalize Results
        print("\n" + "=" * 80)
        print("FINALIZING RESULTS")
        print("=" * 80)

        final_result = {
            "session_name": session_name,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "model": self.model,
                "max_iterations": self.max_iterations,
                "acceptance_threshold": self.acceptance_threshold,
                "expert_criteria": self.expert_criteria
            },
            "generation": {
                "num_chunks": len(generation_result["chunks"]),
                "num_codes": len(codes),
                "initial_num_themes": len(generation_result["themes"])
            },
            "refinement_iterations": iteration,
            "refinement_history": refinement_history,
            "final_themes": themes,
            "final_evaluation": evaluation_result if not is_acceptable and iteration >= self.max_iterations else None,
            "accepted": is_acceptable,
            "metadata": {
                "total_themes": len(themes),
                "final_average_score": evaluation_result["average_score"]
            }
        }

        # Save final results
        final_path = os.path.join(session_dir, "00_final_results.json")
        with open(final_path, 'w') as f:
            json.dump(final_result, f, indent=2)

        print(f"\n✓ Final results saved to: {final_path}")
        print(f"\n  Total themes: {len(themes)}")
        print(f"  Final score: {evaluation_result['average_score']:.2f}/5.0")
        print(f"  Iterations: {iteration}")
        print(f"  Status: {'ACCEPTED' if is_acceptable else 'MAX_ITERATIONS_REACHED'}")

        # Create human-readable summary
        self._save_readable_summary(session_dir, final_result)

        print("\n" + "=" * 80)
        print("TAMA ANALYSIS COMPLETE")
        print("=" * 80)

        return final_result

    def _save_readable_summary(self, session_dir: str, result: Dict[str, Any]):
        """
        Save a human-readable summary of the analysis.

        Args:
            session_dir: Directory to save summary
            result: Final analysis result
        """
        summary_path = os.path.join(session_dir, "00_summary.txt")

        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TAMA THEMATIC ANALYSIS - SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Session: {result['session_name']}\n")
            f.write(f"Timestamp: {result['timestamp']}\n")
            f.write(f"Model: {result['configuration']['model']}\n")
            f.write(f"Status: {'ACCEPTED' if result['accepted'] else 'MAX_ITERATIONS_REACHED'}\n")
            f.write(f"Final Score: {result['metadata']['final_average_score']:.2f}/5.0\n")
            f.write(f"Refinement Iterations: {result['refinement_iterations']}\n\n")

            f.write("=" * 80 + "\n")
            f.write("FINAL THEMES\n")
            f.write("=" * 80 + "\n\n")

            for idx, theme in enumerate(result['final_themes'], 1):
                f.write(f"{idx}. {theme['name']}\n")
                f.write(f"   {theme['description']}\n")
                f.write(f"   Associated codes: {len(theme.get('codes', []))}\n\n")

            if result['refinement_history']:
                f.write("\n" + "=" * 80 + "\n")
                f.write("REFINEMENT HISTORY\n")
                f.write("=" * 80 + "\n\n")

                for iteration_data in result['refinement_history']:
                    iter_num = iteration_data['iteration']
                    f.write(f"Iteration {iter_num}:\n")
                    f.write(f"  Evaluation Score: {iteration_data['evaluation']['average_score']:.2f}/5.0\n")
                    f.write(f"  Refinement: {iteration_data['refinement']['refinement_plan']['summary']}\n")
                    f.write(f"  Theme Count: {iteration_data['refinement']['theme_count_before']} -> {iteration_data['refinement']['theme_count_after']}\n\n")

        print(f"✓ Human-readable summary saved to: {summary_path}")


def load_transcript(file_path: str) -> str:
    """
    Load transcript from a text file.

    Args:
        file_path: Path to transcript file

    Returns:
        Transcript text
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
