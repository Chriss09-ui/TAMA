"""
TAMA: A Human-AI Collaborative Thematic Analysis Framework Using Multi-Agent LLMs
Main orchestrator coordinating Generation, Evaluation, and Refinement agents.
"""

import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import json
import uuid

from agents.generation_agent import GenerationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.refinement_agent import RefinementAgent
from chunking import ChunkStrategy, resolve_chunk_strategy
from decisions.base import DecisionProvider
from prompts import ENTERPRISE_CRITERIA
from research_profile import resolve_profile


def without_excerpts(value):
    """Remove explicit verbatim excerpt fields from a persisted result."""
    if isinstance(value, dict):
        return {key: without_excerpts(item) for key, item in value.items() if key != "excerpt"}
    if isinstance(value, list):
        return [without_excerpts(item) for item in value]
    return value


def link_themes_to_codes(themes, codes):
    """Keep theme references tied to existing codes after model refinement."""
    code_by_id = {code.get("code_id"): code for code in codes if isinstance(code.get("code_id"), int)}
    ids_by_description = {}
    for code_id, code in code_by_id.items():
        ids_by_description.setdefault(code.get("description"), []).append(code_id)
    linked = []
    for original in themes:
        theme = dict(original)
        ids = [code_id for code_id in theme.get("code_ids", []) if code_id in code_by_id]
        if not ids:
            ids = [
                code_id for description in theme.get("codes", [])
                for code_id in ids_by_description.get(description, [])
            ]
        theme["code_ids"] = list(dict.fromkeys(ids))
        theme["counterexample_code_ids"] = list(dict.fromkeys(
            code_id for code_id in theme.get("counterexample_code_ids", [])
            if code_id in code_by_id
        ))
        theme["codes"] = [code_by_id[code_id]["description"] for code_id in theme["code_ids"]]
        theme["open_questions"] = list(theme.get("open_questions") or [])
        theme["kind"] = theme.get("kind") or "pattern"
        if not theme["code_ids"]:
            theme["kind"] = "evidence_gap"
            theme["open_questions"].append("该发现缺少可追溯的原始编码，需人工核对。")
        linked.append(theme)
    return linked


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
        expert_criteria: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
        chunk_size: Optional[int] = None,
        max_workers: int = 4,
        chunk_strategy: Optional[ChunkStrategy] = None,
        decision_provider: Optional[DecisionProvider] = None,
        confidence_threshold: float = 0.7,
        profile: str = "generic",
        research_question: str = "",
        focus_areas: Optional[list[str]] = None,
        early_stop_patience: int = 2,
    ):
        """
        Initialize TAMA Framework.

        Args:
            api_key: API key for the selected model provider
            model: Model to use for all agents (default: gpt-4o)
            max_iterations: Maximum refinement iterations (default: 5)
            acceptance_threshold: Minimum score to accept themes (default: 4.0)
            output_dir: Directory to save outputs (default: outputs)
            expert_criteria: Optional study-specific evaluation criteria from a researcher
            base_url: Optional OpenAI-compatible API endpoint
            chunk_size: Manual maximum Chinese characters or other words per initial chunk
            max_workers: Maximum concurrent code extraction and theme evaluation requests
            chunk_strategy: fine, balanced, economy, or manual. Omitted values use
                manual mode when chunk_size is supplied, otherwise balanced mode.
            decision_provider: Optional typed-decision provider; the main model is used when omitted
            confidence_threshold: Decision confidence below this flags a theme for human review
            profile: generic or enterprise_evidence prompt and rubric profile
            research_question: Optional study question injected into prompts
            focus_areas: Optional focus labels injected into prompts
            early_stop_patience: Consecutive non-improving evaluations before stopping
        """
        self.api_key = api_key
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.model = model
        self.max_iterations = max_iterations
        self.acceptance_threshold = acceptance_threshold
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_strategy = resolve_chunk_strategy(chunk_strategy, chunk_size)
        self.max_workers = max_workers
        self.decision_provider = decision_provider
        self.confidence_threshold = confidence_threshold
        self.study = resolve_profile(profile, research_question, focus_areas)
        if early_stop_patience < 1:
            raise ValueError("early_stop_patience must be positive")
        self.early_stop_patience = early_stop_patience
        criteria = dict(ENTERPRISE_CRITERIA) if profile == "enterprise_evidence" else {}
        criteria.update(expert_criteria or {})
        self.expert_criteria = criteria or None

        # Initialize agents
        self.generation_agent = GenerationAgent(
            api_key=api_key, model=model, base_url=base_url,
            chunk_size=chunk_size, chunk_strategy=self.chunk_strategy,
            max_workers=max_workers,
            study=self.study,
        )
        self.evaluation_agent = EvaluationAgent(
            api_key=api_key,
            model=model,
            expert_criteria=criteria or None,
            base_url=base_url,
            max_workers=max_workers,
            decision_provider=decision_provider,
            confidence_threshold=confidence_threshold,
            study=self.study,
        )
        self.refinement_agent = RefinementAgent(api_key=api_key, model=model, base_url=base_url, study=self.study)

    def run_analysis(
        self,
        transcript: str,
        session_name: str = None,
        save_intermediate: bool = False,
        before_model_call: Optional[Callable[[str], None]] = None,
        case_id: Optional[str] = None,
        save_final: bool = True,
        redact_saved_quotes: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete TAMA analysis with iterative refinement.

        Args:
            transcript: Interview transcript text
            session_name: Optional name for this analysis session
            save_intermediate: Whether to save intermediate results
            before_model_call: Optional checkpoint called before each model request
            case_id: Optional pseudonymous case identifier for research records
            save_final: Write final JSON and summary when true
            redact_saved_quotes: Remove excerpt fields from saved JSON;
                requires save_intermediate=False

        Returns:
            Dictionary containing final themes and analysis metadata
        """
        if redact_saved_quotes and save_intermediate:
            raise ValueError("脱敏保存时不能保存含逐字稿的阶段文件")
        if session_name is None:
            session_name = f"tama_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        if os.path.basename(session_name) != session_name or session_name in (".", ".."):
            raise ValueError("session_name must be a single directory name")

        self.generation_agent.before_model_call = before_model_call
        self.evaluation_agent.before_model_call = before_model_call
        self.refinement_agent.before_model_call = before_model_call

        session_dir = None
        if save_final or save_intermediate:
            os.makedirs(self.output_dir, exist_ok=True)
            candidate = session_name
            while True:
                session_dir = os.path.join(self.output_dir, candidate)
                try:
                    os.mkdir(session_dir)
                    session_name = candidate
                    break
                except FileExistsError:
                    candidate = f"{session_name}_{uuid.uuid4().hex[:8]}"

        print("=" * 80)
        print(f"TAMA FRAMEWORK - Session: {session_name}")
        print("=" * 80)

        # Phase 1: Generation
        print("\n" + "=" * 80)
        print("PHASE 1: GENERATION AGENT")
        print("=" * 80)

        generation_path = os.path.join(session_dir, "01_generation.json") if save_intermediate else None
        generation_result = self.generation_agent.run(transcript, save_path=generation_path)

        themes = link_themes_to_codes(generation_result["themes"], generation_result["codes"])
        codes = generation_result["codes"]

        # Phase 2: Iterative Evaluation and Refinement
        iteration = 0
        is_acceptable = False
        refinement_history = []
        score_history = []
        best_score = float("-inf")
        stagnant_rounds = 0
        stop_reason = "max_iterations"

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
                save_path=eval_path,
                acceptance_threshold=self.acceptance_threshold,
            )

            print(f"\nEvaluation Results:")
            print(f"  Average Score: {evaluation_result['average_score']:.2f}/5.0")
            print(f"  Acceptable: {evaluation_result['is_acceptable']}")
            print(f"  Feedback: {evaluation_result['global_feedback']}")

            is_acceptable = evaluation_result["is_acceptable"]
            score = evaluation_result["average_score"]
            score_history.append({"iteration": iteration, "average_score": score})
            if score > best_score + 0.05:
                best_score = score
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1

            # If acceptable or max iterations reached, stop
            if is_acceptable:
                stop_reason = "accepted"
                print("\n✓ Themes are acceptable! Analysis complete.")
                break

            if stagnant_rounds >= self.early_stop_patience:
                stop_reason = "no_improvement"
                print("\n⚠ Scores have stopped improving; ending refinement.")
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
            themes = link_themes_to_codes(refinement_result["refined_themes"], codes)

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
            "case_id": case_id.strip() if case_id else None,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "model": self.model,
                "max_iterations": self.max_iterations,
                "chunk_size": self.chunk_size,
                "chunk_strategy": self.chunk_strategy,
                "max_workers": self.max_workers,
                "acceptance_threshold": self.acceptance_threshold,
                "expert_criteria": self.expert_criteria,
                "decision_provider": (
                    self.decision_provider.name
                    if self.decision_provider else f"llm:{self.model}"
                ),
                "confidence_threshold": self.confidence_threshold,
                "research_profile": self.study.name,
                "research_question": self.study.research_question,
                "focus_areas": list(self.study.focus_areas),
                "early_stop_patience": self.early_stop_patience,
                "save_final": save_final,
                "save_intermediate": save_intermediate,
                "redact_saved_quotes": redact_saved_quotes,
            },
            "generation": {
                "num_chunks": len(generation_result["chunks"]),
                "num_codes": len(codes),
                "initial_num_themes": len(generation_result["themes"]),
                "chunking": generation_result.get("chunking"),
                "analytic_storyline": generation_result.get("analytic_storyline", ""),
            },
            "refinement_iterations": iteration,
            "score_history": score_history,
            "stop_reason": stop_reason,
            "refinement_history": refinement_history,
            "codes": codes,
            "final_themes": themes,
            "final_evaluation": evaluation_result,
            "accepted": is_acceptable,
            "metadata": {
                "total_themes": len(themes),
                "final_average_score": evaluation_result["average_score"]
            }
        }

        if save_final:
            final_path = os.path.join(session_dir, "00_final_results.json")
            persisted = without_excerpts(final_result) if redact_saved_quotes else final_result
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(persisted, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Final results saved to: {final_path}")
        print(f"\n  Total themes: {len(themes)}")
        print(f"  Final score: {evaluation_result['average_score']:.2f}/5.0")
        print(f"  Iterations: {iteration}")
        print(f"  Status: {stop_reason.upper()}")

        if save_final:
            self._save_readable_summary(session_dir, persisted)

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
            f.write(f"Status: {result['stop_reason'].upper()}\n")
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
