"""
Threadline: Human-AI collaborative qualitative analysis
Main orchestrator coordinating Generation, Evaluation, and Refinement agents.
"""

import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import json
import uuid
from dataclasses import replace

from agents.generation_agent import GenerationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.refinement_agent import RefinementAgent
from chunking import ChunkStrategy, resolve_chunk_strategy
from code_mapping import attach_category_names, render_code_landscape, render_code_map
from code_memos import render_code_memos_markdown
from codebook import render_codebook
from decisions.base import DecisionProvider
from memos import (
    attach_operation_rationales,
    build_theme_memos,
    memos_for_model,
    render_memos_markdown,
)
from next_data_plan import build_next_data_plan
from evidence import link_themes_to_codes, matched_codes, source_document, validate_documents, validate_evidence
from prompts import ENTERPRISE_CRITERIA
from reporting import render_summary_text
from research_profile import resolve_profile
from research_records import (
    AnalyticDecision, SourceMetadata, build_workspace, continuation_context,
    decisions_for_model, prepare_previous_result, render_research_records,
)


class ThreadlineFramework:
    """
    Threadline orchestrator that coordinates multi-agent workflow:
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
        analysis_mode: str = "thematic",
        corpus_review: bool = True,
    ):
        """
        Initialize Threadline Framework.

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
            corpus_review: Check the submitted full text once before scoring themes
            early_stop_patience: Consecutive non-improving evaluations before stopping
        """
        self.api_key = api_key
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not isinstance(corpus_review, bool):
            raise ValueError("corpus_review must be a boolean")
        self.model = model
        self.max_iterations = max_iterations
        self.corpus_review = corpus_review
        self.acceptance_threshold = acceptance_threshold
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_strategy = resolve_chunk_strategy(chunk_strategy, chunk_size)
        self.max_workers = max_workers
        self.decision_provider = decision_provider
        self.confidence_threshold = confidence_threshold
        if analysis_mode not in ("thematic", "grounded_theory"):
            raise ValueError("Unknown analysis mode")
        self.study = replace(resolve_profile(profile, research_question, focus_areas), analysis_mode=analysis_mode)
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
        source_metadata: Optional[Dict[str, Any]] = None,
        previous_result: Optional[Dict[str, Any]] = None,
        confirmed_decisions: Optional[list[dict]] = None,
        include_previous_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete Threadline analysis with iterative refinement.

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
            session_name = f"threadline_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        if os.path.basename(session_name) != session_name or session_name in (".", ".."):
            raise ValueError("session_name must be a single directory name")

        previous_codes, previous_workspace = [], None
        if previous_result is not None:
            previous_codes, previous_workspace = prepare_previous_result(previous_result)
        decisions = [AnalyticDecision.model_validate(item).model_dump() for item in confirmed_decisions or []]
        prior_ids = {code["code_id"] for code in previous_codes}
        if any(code_id not in prior_ids for decision in decisions for code_id in decision["code_ids"]):
            raise ValueError("分析决定只能引用已载入的前轮编码")
        source = SourceMetadata.model_validate(source_metadata or {})
        if case_id and not source.case_id:
            source.case_id = case_id.strip()
        source.source_id = source.source_id or uuid.uuid4().hex
        documents = validate_documents((previous_result or {}).get("source_documents") or [])
        documents.append(source_document(source.source_id, transcript))
        if any((code.get("source") or {}).get("source_id") == source.source_id for code in previous_codes):
            raise ValueError("新资料编号与前轮重复，请为这份新资料使用独立编号")
        current_study = replace(
            self.study, confirmed_decisions=decisions_for_model(decisions),
            previous_analytic_context=continuation_context(previous_workspace)
                if previous_workspace is not None and include_previous_context else "",
        )
        for agent in (self.generation_agent, self.evaluation_agent, self.refinement_agent):
            agent.study = current_study

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
        print(f"THREADLINE - Session: {session_name}")
        print("=" * 80)

        # Phase 1: Generation
        print("\n" + "=" * 80)
        print("PHASE 1: GENERATION AGENT")
        print("=" * 80)

        generation_path = os.path.join(session_dir, "01_generation.json") if save_intermediate else None
        generation_result = self.generation_agent.run(
            transcript, save_path=generation_path, source_metadata=source.model_dump(),
            previous_codes=previous_codes, confirmed_decisions=decisions,
            source_documents=documents, corpus_review=self.corpus_review,
        )
        current_storyline = generation_result.get("analytic_storyline") or ""

        codes = generation_result["codes"]
        validate_evidence(codes, documents)
        themes = link_themes_to_codes(generation_result["themes"], codes)
        review = generation_result.get("corpus_review") or {"status": "disabled", "findings": []}
        code_map = generation_result.get("code_map")
        themes = attach_category_names(themes, codes, code_map)
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
        retired_findings = []
        evaluation_result = {"theme_evaluations": [], "average_score": None,
                             "is_acceptable": False, "global_feedback": "没有可评估的有效证据或主题；未评分。"}
        eligible = matched_codes(codes)
        while eligible and themes and not is_acceptable and iteration < self.max_iterations:
            if not any(theme.get("code_ids") and theme.get("kind") != "evidence_gap" for theme in themes):
                break
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
                codes=eligible,
                save_path=eval_path,
                acceptance_threshold=self.acceptance_threshold,
            )

            print(f"\nEvaluation Results:")
            print(f"  Average Score: {evaluation_result['average_score']:.2f}/5.0")
            print(f"  Acceptable: {evaluation_result['is_acceptable']}")
            print(f"  Feedback: {evaluation_result['global_feedback']}")

            is_acceptable = evaluation_result["is_acceptable"] and all(
                theme.get("code_ids") and theme.get("kind") != "evidence_gap" for theme in themes)
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
                print("\n✓ This round's themes meet the evaluation criteria.")
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
                codes=eligible,
                save_path=refinement_path,
                memos=memos_for_model(memos),
            )
            old_by_name = {theme["name"]: theme for theme in themes}
            for operation in refinement_result.get("refinement_plan", {}).get("operations") or []:
                if operation.get("operation") == "delete":
                    for name in operation.get("target_themes") or []:
                        old = old_by_name.get(name)
                        if old and (old.get("counterexample_code_ids") or old.get("open_questions") or old.get("uncertain")
                                    or old.get("kind") in ("counterexample", "evidence_gap")):
                            retired_findings.append({"theme": old, "reason": operation.get("rationale") or "",
                                                     "iteration": iteration})
            if refinement_result.get("refined_themes") != themes:
                current_storyline = refinement_result.get("analytic_storyline") or ""

            # Update themes for next iteration
            themes = link_themes_to_codes(refinement_result["refined_themes"], codes)
            themes = attach_category_names(themes, codes, code_map)
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

        if evaluation_result["average_score"] is None:
            stop_reason = "no_valid_evidence" if not eligible else "no_valid_themes"
        if self.corpus_review and review["status"] != "complete":
            is_acceptable = False
            stop_reason = "corpus_review_incomplete"
        if before_model_call:
            before_model_call("整理本次结果")

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
                "analysis_mode": self.study.analysis_mode,
                "corpus_review": self.corpus_review,
            },
            "generation": {
                "num_chunks": len(generation_result["chunks"]),
                "num_codes": len(codes),
                "initial_num_themes": generation_result.get("initial_num_themes", len(generation_result["themes"])),
                "chunking": generation_result.get("chunking"),
                "analytic_storyline": generation_result.get("analytic_storyline", ""),
                "codebook_note": generation_result.get("codebook_note", ""),
                "code_map": code_map,
            },
            "code_map": code_map,
            "code_memos": generation_result.get("code_memos") or [],
            "refinement_iterations": iteration,
            "score_history": score_history,
            "stop_reason": stop_reason,
            "refinement_history": refinement_history,
            "codes": codes,
            "source_documents": documents,
            "corpus_review": review,
            "final_themes": themes,
            "memos": memos,
            "next_data_plan": build_next_data_plan(codes, themes, memos),
            "analytic_storyline": current_storyline,
            "storyline_needs_review": bool(refinement_history) and not bool(current_storyline),
            "retired_findings": retired_findings,
            "research_workspace": build_workspace(
                codes, code_map, session_name, source, previous_workspace, decisions,
            ),
            "final_evaluation": evaluation_result,
            "accepted": is_acceptable,
            "metadata": {
                "total_themes": len(themes),
                "valid_code_count": len(eligible),
                "pending_code_count": len(codes) - len(eligible),
                "final_average_score": evaluation_result["average_score"]
            }
        }
        final_result["next_data_plan"] = build_next_data_plan(
            codes, [*themes, *(item["theme"] for item in retired_findings)], memos,
            final_result["research_workspace"], review,
        )
        if session_dir:
            final_result["output_dir"] = session_dir

        if save_final:
            final_path = os.path.join(session_dir, "00_final_results.json")
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Final results saved to: {final_path}")
        print(f"\n  Total themes: {len(themes)}")
        score = evaluation_result["average_score"]
        print(f"  Final score: {score if score is not None else 'not scored'}")
        print(f"  Iterations: {iteration}")
        print(f"  Status: {stop_reason.upper()}")

        if save_final:
            self._save_readable_summary(session_dir, final_result)
        if session_dir:
            self._write_research_records(session_dir, final_result)

        print("\n" + "=" * 80)
        print("THREADLINE ANALYSIS COMPLETE")
        print("=" * 80)

        return final_result

    def _write_memo_iteration(self, session_dir, memos, iteration):
        if not session_dir:
            return
        path = os.path.join(session_dir, f"04_memos_iter{iteration}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(memos_for_model(memos), handle, indent=2, ensure_ascii=False)

    def _write_research_records(self, session_dir, result):
        memos = result.get("memos") or []
        codes = result.get("codes") or []
        plan = result.get("next_data_plan") or {}
        code_map = result.get("code_map")
        memo_path = os.path.join(session_dir, "04_memos.md")
        with open(memo_path, "w", encoding="utf-8") as handle:
            handle.write(render_memos_markdown(memos))
        with open(os.path.join(session_dir, "04_memos.json"), "w", encoding="utf-8") as handle:
            json.dump(memos_for_model(memos), handle, indent=2, ensure_ascii=False)
        with open(os.path.join(session_dir, "04_code_memos.md"), "w", encoding="utf-8") as handle:
            handle.write(render_code_memos_markdown(result.get("code_memos") or []))
        with open(os.path.join(session_dir, "05_codebook.md"), "w", encoding="utf-8") as handle:
            handle.write(render_codebook(codes))
        with open(os.path.join(session_dir, "06_code_map.md"), "w", encoding="utf-8") as handle:
            handle.write(render_code_map(code_map, matched_codes(codes)))
        with open(os.path.join(session_dir, "07_code_landscape.md"), "w", encoding="utf-8") as handle:
            handle.write(render_code_landscape(matched_codes(codes), code_map))
        with open(os.path.join(session_dir, "next_data_plan.md"), "w", encoding="utf-8") as handle:
            handle.write(plan.get("markdown") or "")
        with open(os.path.join(session_dir, "08_research_records.md"), "w", encoding="utf-8") as handle:
            handle.write(render_research_records(result["research_workspace"]))
        print(f"✓ Memos, codebook, code map, and material checks saved to: {session_dir}")

    def _save_readable_summary(self, session_dir: str, result: Dict[str, Any]):
        """Save a human-readable summary of the analysis."""
        summary_path = os.path.join(session_dir, "00_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(render_summary_text(result))
        print(f"✓ Human-readable summary saved to: {summary_path}")


# Preserve the import used by scripts written before the project rename.
TAMAFramework = ThreadlineFramework


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
