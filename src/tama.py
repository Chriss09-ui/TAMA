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
from codebook import render_codebook
from decisions.base import DecisionProvider
from memos import (
    attach_operation_rationales,
    build_theme_memos,
    memos_for_model,
    render_memos_markdown,
)
from next_data_plan import build_next_data_plan
from prompts import ENTERPRISE_CRITERIA
from reporting import render_summary_text
from research_profile import resolve_profile


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

        Returns:
            Dictionary containing final themes and analysis metadata
        """
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
        memos = build_theme_memos(themes, iteration=0)
        self._write_memo_iteration(session_dir, memos, 0)

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
                save_path=refinement_path,
                memos=memos_for_model(memos),
            )

            # Update themes for next iteration
            themes = link_themes_to_codes(refinement_result["refined_themes"], codes)
            attach_operation_rationales(
                themes, refinement_result.get("refinement_plan", {}).get("operations"),
            )
            memos = build_theme_memos(themes, previous=memos, iteration=iteration)
            self._write_memo_iteration(session_dir, memos, iteration)

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
            },
            "generation": {
                "num_chunks": len(generation_result["chunks"]),
                "num_codes": len(codes),
                "initial_num_themes": len(generation_result["themes"]),
                "chunking": generation_result.get("chunking"),
                "analytic_storyline": generation_result.get("analytic_storyline", ""),
                "codebook_note": generation_result.get("codebook_note", ""),
            },
            "refinement_iterations": iteration,
            "score_history": score_history,
            "stop_reason": stop_reason,
            "refinement_history": refinement_history,
            "codes": codes,
            "final_themes": themes,
            "memos": memos,
            "next_data_plan": build_next_data_plan(codes, themes, memos),
            "final_evaluation": evaluation_result,
            "accepted": is_acceptable,
            "metadata": {
                "total_themes": len(themes),
                "final_average_score": evaluation_result["average_score"]
            }
        }
        if session_dir:
            final_result["output_dir"] = session_dir

        if save_final:
            final_path = os.path.join(session_dir, "00_final_results.json")
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Final results saved to: {final_path}")
        print(f"\n  Total themes: {len(themes)}")
        print(f"  Final score: {evaluation_result['average_score']:.2f}/5.0")
        print(f"  Iterations: {iteration}")
        print(f"  Status: {stop_reason.upper()}")

        if save_final:
            self._save_readable_summary(session_dir, final_result)
        if session_dir:
            self._write_research_records(
                session_dir, memos, codes, final_result["next_data_plan"],
            )

        print("\n" + "=" * 80)
        print("TAMA ANALYSIS COMPLETE")
        print("=" * 80)

        return final_result

    def _write_memo_iteration(self, session_dir, memos, iteration):
        if not session_dir:
            return
        path = os.path.join(session_dir, f"04_memos_iter{iteration}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(memos_for_model(memos), handle, indent=2, ensure_ascii=False)

    def _write_research_records(self, session_dir, memos, codes, plan):
        memo_path = os.path.join(session_dir, "04_memos.md")
        with open(memo_path, "w", encoding="utf-8") as handle:
            handle.write(render_memos_markdown(memos))
        with open(os.path.join(session_dir, "04_memos.json"), "w", encoding="utf-8") as handle:
            json.dump(memos_for_model(memos), handle, indent=2, ensure_ascii=False)
        with open(os.path.join(session_dir, "05_codebook.md"), "w", encoding="utf-8") as handle:
            handle.write(render_codebook(codes))
        with open(os.path.join(session_dir, "next_data_plan.md"), "w", encoding="utf-8") as handle:
            handle.write(plan["markdown"])
        print(f"✓ Memos, codebook, and next-round plan saved to: {session_dir}")

    def _save_readable_summary(self, session_dir: str, result: Dict[str, Any]):
        """Save a human-readable summary of the analysis."""
        summary_path = os.path.join(session_dir, "00_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(render_summary_text(result))
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
