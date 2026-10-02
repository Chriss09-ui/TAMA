"""
Evaluation Agent for Threadline Framework
Evaluates generated themes based on four criteria: Coverage, Actionability, Distinctiveness, and Relevance.
Provides feedback for refinement until affirmative answer is received.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, Field, TypeAdapter, field_validator
import json
import math
from concurrent.futures import ThreadPoolExecutor

from decisions.base import DecisionProvider, DecisionQuestion
from decisions.llm_client import LLMDecisionClient
from prompts import (
    DEFAULT_ACTIONABILITY_CRITERION,
    DEFAULT_COVERAGE_CRITERION,
    DEFAULT_DISTINCTIVENESS_CRITERION,
    DEFAULT_RELEVANCE_CRITERION,
    CRITERION_SCALES,
    EVALUATION_FEEDBACK_SYSTEM_PROMPT,
    FULL_EVALUATION_SYSTEM_PROMPT,
    build_evaluation_feedback_prompt,
    build_evaluation_question_instructions,
    build_full_evaluation_prompt,
)
from research_profile import ResearchProfile
from evidence import link_themes_to_codes, matched_codes
from analysis_records import scorable_themes


class EvaluationCriteria(BaseModel):
    """Represents evaluation criteria for themes."""
    coverage: str = DEFAULT_COVERAGE_CRITERION
    actionability: str = DEFAULT_ACTIONABILITY_CRITERION
    distinctiveness: str = DEFAULT_DISTINCTIVENESS_CRITERION
    relevance: str = DEFAULT_RELEVANCE_CRITERION


class EvaluationResult(BaseModel):
    """Represents evaluation result for a theme."""
    theme_name: str
    coverage_score: int = Field(ge=1, le=5)
    coverage_feedback: str
    actionability_score: int = Field(ge=1, le=5)
    actionability_feedback: str
    distinctiveness_score: int = Field(ge=1, le=5)
    distinctiveness_feedback: str
    relevance_score: int = Field(ge=1, le=5)
    relevance_feedback: str
    overall_score: float = Field(ge=1, le=5, allow_inf_nan=False)
    needs_refinement: bool
    refinement_suggestions: List[str]
    # Decision-layer audit fields (populated when a decision provider scored
    # the theme; None keeps legacy single-call results unchanged)
    score_confidences: Optional[Dict[str, float]] = None
    needs_refinement_confidence: Optional[float] = None
    raw_scores: Optional[Dict[str, float]] = None
    weighted_scores: Optional[Dict[str, float]] = None
    flagged_for_review: bool = False
    feedback_source: str = "llm"  # "llm" | "placeholder" | "fallback"

    @field_validator("coverage_score", "actionability_score", "distinctiveness_score", "relevance_score", mode="before")
    @classmethod
    def numeric_rating(cls, value):
        if type(value) not in (int, float):
            raise ValueError("评估分数必须为 1–5 的整数")
        return value


class OverallEvaluation(BaseModel):
    """Overall evaluation of all themes."""
    theme_evaluations: List[EvaluationResult]
    average_score: Optional[float]
    is_acceptable: bool
    global_feedback: str


CRITERION_KEYS = ("coverage", "actionability", "distinctiveness", "relevance")


def criterion_score(evaluation: EvaluationResult, key: str) -> float:
    weighted_scores = getattr(evaluation, "weighted_scores", None)
    if weighted_scores is not None:
        return weighted_scores[key]
    return float(getattr(evaluation, f"{key}_score"))


class EvaluationAgent:
    """
    Evaluation Agent that assesses themes based on four criteria:
    1. Coverage: Comprehensively captures important patterns
    2. Actionability: Encapsulates single, clear concept
    3. Distinctiveness: Clearly distinct from other themes
    4. Relevance: Accurately reflects the supplied data

    Provides feedback until affirmative answer is received.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        expert_criteria: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
        max_workers: int = 4,
        decision_provider: Optional[DecisionProvider] = None,
        confidence_threshold: float = 0.7,
        study: Optional[ResearchProfile] = None,
    ):
        """
        Initialize the Evaluation Agent.

        Args:
            api_key: API key for the selected model provider
            model: Model to use (default: gpt-4o)
            expert_criteria: Optional study-specific evaluation criteria from a researcher
            base_url: Optional OpenAI-compatible API endpoint
            max_workers: Maximum concurrent theme evaluation requests
            decision_provider: Optional typed-decision provider for hybrid scoring
            confidence_threshold: Decision confidence below this flags a theme for human review
        """
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0, 1]")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_workers = max_workers
        if decision_provider is None:
            # Default decision path: the main model answers typed questions
            # in JSON mode, so hybrid evaluation works with every provider.
            decision_provider = LLMDecisionClient(api_key=api_key, model=model, base_url=base_url)
        self.decision_provider = decision_provider
        self.confidence_threshold = confidence_threshold
        self.before_model_call = None
        self.study = study or ResearchProfile()

        # Use expert-provided criteria or defaults
        if expert_criteria:
            self.criteria = EvaluationCriteria(**expert_criteria)
        else:
            self.criteria = EvaluationCriteria()

    def evaluate_theme(
        self,
        theme: Dict[str, Any],
        all_themes: List[Dict[str, Any]],
        original_codes: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate a single theme against the four criteria.

        With a decision provider, scores come from typed decisions and the
        detailed feedback call only runs for themes that need attention.

        Args:
            theme: Theme dictionary to evaluate
            all_themes: List of all themes for distinctiveness check
            original_codes: Original codes for coverage check

        Returns:
            EvaluationResult object
        """
        theme = link_themes_to_codes([theme], original_codes)[0]
        all_themes = link_themes_to_codes(all_themes, original_codes)
        if not theme.get("code_ids") or theme.get("kind") == "evidence_gap":
            raise ValueError("主题没有可评分的原文支持")
        if self.decision_provider is not None:
            return self._evaluate_theme_with_decisions(theme, all_themes, self._compact_codes(theme, original_codes))
        return self._evaluate_theme_full(theme, all_themes, self._compact_codes(theme, original_codes))

    @staticmethod
    def _compact_codes(theme, codes):
        linked = set([*(theme.get("code_ids") or []), *(theme.get("counterexample_code_ids") or [])])
        return [
            {
                "description": code.get("description"),
                **({"code_id": code["code_id"]} if "code_id" in code else {}),
                **({"excerpt": str(code.get("excerpt"))[:180]} if code.get("code_id") in linked and code.get("excerpt") else {}),
                **({"verification_status": code.get("verification_status")} if code.get("code_id") in linked else {}),
                **({"statement_type": code.get("statement_type")} if code.get("code_id") in linked and code.get("statement_type") else {}),
                **({"focus": code.get("focus")} if code.get("code_id") in linked and code.get("focus") else {}),
                **({"open_question": code.get("open_question")} if code.get("code_id") in linked and code.get("open_question") else {}),
                **({"source": code.get("source") or {}, "context": code.get("context") or "", "note": code.get("note") or ""}
                   if code.get("code_id") in linked else {}),
            }
            for code in matched_codes(codes)
        ]

    def _evaluate_theme_with_decisions(
        self,
        theme: Dict[str, Any],
        all_themes: List[Dict[str, Any]],
        original_codes: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Score the theme with typed decisions; write feedback only when needed."""
        other_themes = [t for t in all_themes if t["name"] != theme["name"]]
        state = {
            "theme": theme,
            "other_themes": other_themes,
            "original_codes": original_codes,
            "research_focus": self.study.prompt_section(),
        }
        instructions = build_evaluation_question_instructions(self.criteria)
        questions = {
            key: DecisionQuestion(
                key=key, kind="score", scale=list(CRITERION_SCALES[key]),
                instructions=instructions[key],
            )
            for key in ("coverage", "actionability", "distinctiveness", "relevance")
        }
        questions["needs_refinement"] = DecisionQuestion(
            key="needs_refinement", kind="noul",
            instructions=instructions["needs_refinement"],
        )

        if self.before_model_call:
            self.before_model_call(f"评估主题 · {theme['name']}")
        try:
            answers = self.decision_provider.ask(state, questions)
        except Exception as exc:
            if getattr(exc, "permanent", False):
                raise
            print(f"  决策调用失败，回退到完整评估（{type(exc).__name__}）")
            fallback = self._evaluate_theme_full(
                theme, all_themes, original_codes,
                stage=f"完整评估 · {theme['name']}",
            )
            fallback.feedback_source = "fallback"
            return fallback

        score_keys = ("coverage", "actionability", "distinctiveness", "relevance")
        scores = {}
        raw_scores = {}
        weighted_scores = {}
        confidences = {}
        for key in score_keys:
            raw = answers[key].score if answers[key].score is not None else 0.0
            scale_size = len(questions[key].scale or CRITERION_SCALES[key])
            bounded_raw = min(float(scale_size - 1), max(0.0, raw))
            # Answers carry a zero-based weighted level index; convert half-up
            # for legacy integer fields, while keeping the continuous 1-5
            # value for branching and aggregate evaluation.
            scores[key] = min(5, max(1, math.floor(bounded_raw + 0.5) + 1))
            raw_scores[key] = raw
            weighted_scores[key] = bounded_raw + 1.0
            confidences[key] = answers[key].confidence
        needs_refinement = (answers["needs_refinement"].probability or 0.0) >= 0.5
        needs_refinement_confidence = answers["needs_refinement"].confidence
        confidences["needs_refinement"] = needs_refinement_confidence
        flagged_for_review = any(
            value < self.confidence_threshold for value in confidences.values()
        )
        overall_score = sum(weighted_scores.values()) / 4.0

        needs_detail = (
            needs_refinement
            or any(weighted_scores[key] < 4.0 for key in score_keys)
            or flagged_for_review
        )
        if needs_detail:
            if self.before_model_call:
                self.before_model_call(f"生成评估反馈 · {theme['name']}")
            feedback = self._generate_detailed_feedback(
                theme, other_themes, original_codes, weighted_scores,
            )
            # The prose model can add a concern, but must not cancel a typed
            # decision or a criterion that scored below the threshold.
            needs_refinement = (
                needs_refinement
                or feedback["needs_refinement"]
                or any(weighted_scores[key] < 4.0 for key in score_keys)
            )
            feedback_source = "llm"
        else:
            feedback = {
                key: f"该项概率加权评分 {weighted_scores[key]:.2f}/5，达到标准；快速评估未生成详细反馈。"
                for key in score_keys
            }
            feedback["refinement_suggestions"] = []
            needs_refinement = False
            feedback_source = "placeholder"

        return EvaluationResult(
            theme_name=theme["name"],
            coverage_score=scores["coverage"],
            coverage_feedback=feedback["coverage"],
            actionability_score=scores["actionability"],
            actionability_feedback=feedback["actionability"],
            distinctiveness_score=scores["distinctiveness"],
            distinctiveness_feedback=feedback["distinctiveness"],
            relevance_score=scores["relevance"],
            relevance_feedback=feedback["relevance"],
            overall_score=overall_score,
            needs_refinement=needs_refinement,
            refinement_suggestions=feedback["refinement_suggestions"],
            score_confidences=confidences,
            needs_refinement_confidence=needs_refinement_confidence,
            raw_scores=raw_scores,
            weighted_scores=weighted_scores,
            flagged_for_review=flagged_for_review,
            feedback_source=feedback_source,
        )

    def _generate_detailed_feedback(
        self,
        theme: Dict[str, Any],
        other_themes: List[Dict[str, Any]],
        original_codes: List[Dict[str, Any]],
        scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Ask the main model for feedback text on already-scored criteria."""
        prompt = build_evaluation_feedback_prompt(
            theme=theme,
            other_themes=other_themes,
            original_codes=original_codes,
            scores=scores,
            study=self.study,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EVALUATION_FEEDBACK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "coverage": result["coverage_feedback"],
            "actionability": result["actionability_feedback"],
            "distinctiveness": result["distinctiveness_feedback"],
            "relevance": result["relevance_feedback"],
            "needs_refinement": TypeAdapter(bool).validate_python(result["needs_refinement"]),
            "refinement_suggestions": result.get("refinement_suggestions", []),
        }

    def _evaluate_theme_full(
        self,
        theme: Dict[str, Any],
        all_themes: List[Dict[str, Any]],
        original_codes: List[Dict[str, Any]],
        stage: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Legacy single-call evaluation; also the fallback path when the
        decision provider fails with a recoverable error.

        Args:
            theme: Theme dictionary to evaluate
            all_themes: List of all themes for distinctiveness check
            original_codes: Original codes for coverage check

        Returns:
            EvaluationResult object
        """
        other_themes = [t for t in all_themes if t["name"] != theme["name"]]
        prompt = build_full_evaluation_prompt(
            criteria=self.criteria,
            theme=theme,
            other_themes=other_themes,
            original_codes=original_codes,
            study=self.study,
        )

        if self.before_model_call:
            self.before_model_call(stage or f"评估主题 · {theme['name']}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FULL_EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Calculate overall score
        overall_score = (
            result["coverage_score"] +
            result["actionability_score"] +
            result["distinctiveness_score"] +
            result["relevance_score"]
        ) / 4.0

        return EvaluationResult(
            theme_name=theme["name"],
            coverage_score=result["coverage_score"],
            coverage_feedback=result["coverage_feedback"],
            actionability_score=result["actionability_score"],
            actionability_feedback=result["actionability_feedback"],
            distinctiveness_score=result["distinctiveness_score"],
            distinctiveness_feedback=result["distinctiveness_feedback"],
            relevance_score=result["relevance_score"],
            relevance_feedback=result["relevance_feedback"],
            overall_score=overall_score,
            needs_refinement=result["needs_refinement"],
            refinement_suggestions=result["refinement_suggestions"]
        )

    def evaluate_all_themes(
        self,
        themes: List[Dict[str, Any]],
        original_codes: List[Dict[str, Any]],
        acceptance_threshold: float = 4.0
    ) -> OverallEvaluation:
        """
        Evaluate all themes and determine if they are acceptable.

        Args:
            themes: List of theme dictionaries
            original_codes: Original codes for coverage check
            acceptance_threshold: Minimum average score to accept (default: 4.0)

        Returns:
            OverallEvaluation object
        """
        print("\nEvaluating themes...")
        if not themes:
            raise ValueError("没有可评估的主题；请检查主题生成结果。")
        original_codes = matched_codes(original_codes)
        themes = link_themes_to_codes(themes, original_codes)
        supported = scorable_themes(themes)
        if not supported:
            return OverallEvaluation(theme_evaluations=[], average_score=None, is_acceptable=False,
                                     global_feedback="没有可按共享模式评分的有效主题；反例与核查发现仍保留供复核；未评分。")

        def evaluate_indexed(item):
            idx, theme = item
            print(f"  Evaluating theme {idx + 1}/{len(themes)}: {theme['name']}")
            return self.evaluate_theme(theme, themes, original_codes)

        if len(supported) > 1 and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(supported))) as executor:
                theme_evaluations = list(executor.map(evaluate_indexed, enumerate(supported)))
        else:
            theme_evaluations = [evaluate_indexed(item) for item in enumerate(supported)]

        # Calculate average score
        average_score = sum(e.overall_score for e in theme_evaluations) / len(theme_evaluations)
        is_acceptable = all(theme.get("code_ids") and theme.get("kind") != "evidence_gap" for theme in themes) and average_score >= acceptance_threshold and all(
            not evaluation.needs_refinement
            and all(criterion_score(evaluation, key) >= acceptance_threshold for key in CRITERION_KEYS)
            for evaluation in theme_evaluations
        )

        # Generate global feedback
        global_feedback = self._generate_global_feedback(
            theme_evaluations, average_score, is_acceptable, acceptance_threshold,
        )

        overall_eval = OverallEvaluation(
            theme_evaluations=theme_evaluations,
            average_score=average_score,
            is_acceptable=is_acceptable,
            global_feedback=global_feedback
        )

        return overall_eval

    def _generate_global_feedback(
        self,
        evaluations: List[EvaluationResult],
        average_score: float,
        is_acceptable: bool,
        acceptance_threshold: float = 4.0,
    ) -> str:
        """
        Generate global feedback summarizing the evaluation.

        Args:
            evaluations: List of theme evaluations
            average_score: Average score across all themes
            is_acceptable: Whether themes are acceptable

        Returns:
            Global feedback string
        """
        themes_needing_refinement = [
            evaluation for evaluation in evaluations
            if evaluation.needs_refinement
            or any(criterion_score(evaluation, key) < acceptance_threshold for key in CRITERION_KEYS)
        ]

        if is_acceptable:
            feedback = f"已达标：主题平均分为 {average_score:.2f}/5.0，达到验收标准。"
            if themes_needing_refinement:
                feedback += f"仍有 {len(themes_needing_refinement)} 个主题可进一步完善。"
        else:
            feedback = f"需要修订：主题平均分为 {average_score:.2f}/5.0，尚未满足逐项验收标准。"
            feedback += f"其中 {len(themes_needing_refinement)} 个主题需要修订。"

            # Summarize common issues
            coverage_issues = sum(1 for e in evaluations if criterion_score(e, "coverage") < acceptance_threshold)
            actionability_issues = sum(1 for e in evaluations if criterion_score(e, "actionability") < acceptance_threshold)
            distinctiveness_issues = sum(1 for e in evaluations if criterion_score(e, "distinctiveness") < acceptance_threshold)
            relevance_issues = sum(1 for e in evaluations if criterion_score(e, "relevance") < acceptance_threshold)

            issues = []
            if coverage_issues > 0:
                issues.append(f"覆盖度（{coverage_issues} 个主题）")
            if actionability_issues > 0:
                issues.append(f"模式性（{actionability_issues} 个主题）")
            if distinctiveness_issues > 0:
                issues.append(f"区分度（{distinctiveness_issues} 个主题）")
            if relevance_issues > 0:
                issues.append(f"相关性（{relevance_issues} 个主题）")

            if issues:
                feedback += f"常见问题：{'、'.join(issues)}。"

        flagged = [e for e in evaluations if getattr(e, "flagged_for_review", False)]
        if flagged:
            feedback += f"有 {len(flagged)} 个主题存在低置信度评分，建议人工复核。"

        return feedback

    def run(
        self,
        themes: List[Dict[str, Any]],
        codes: List[Dict[str, Any]],
        save_path: str = None,
        acceptance_threshold: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Run the evaluation process on generated themes.

        Args:
            themes: List of theme dictionaries
            codes: List of code dictionaries
            save_path: Optional path to save evaluation results

        Returns:
            Dictionary containing evaluation results
        """
        codes = matched_codes(codes)
        themes = link_themes_to_codes(themes, codes)
        overall_eval = self.evaluate_all_themes(themes, codes, acceptance_threshold)
        if any(not theme.get("code_ids") or theme.get("kind") == "evidence_gap" for theme in themes):
            overall_eval.is_acceptable = False

        result = {
            "theme_evaluations": [e.model_dump() for e in overall_eval.theme_evaluations],
            "average_score": overall_eval.average_score,
            "is_acceptable": overall_eval.is_acceptable,
            "global_feedback": overall_eval.global_feedback,
            "decision_provider": self.decision_provider.name if self.decision_provider else None,
            "confidence_threshold": self.confidence_threshold,
            "flagged_themes": [
                e.theme_name for e in overall_eval.theme_evaluations if e.flagged_for_review
            ],
        }

        # Save evaluation results if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved evaluation results to {save_path}")

        return result
