"""
Evaluation Agent for TAMA Framework
Evaluates generated themes based on four criteria: Coverage, Actionability, Distinctiveness, and Relevance.
Provides feedback for refinement until affirmative answer is received.
"""

from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel
import json


class EvaluationCriteria(BaseModel):
    """Represents evaluation criteria for themes."""
    coverage: str = "should comprehensively capture all important patterns and concepts from the data"
    actionability: str = "should encapsulate a single concept that is clear and actionable"
    distinctiveness: str = "should be clearly distinct from other themes without overlap"
    relevance: str = "must accurately reflect the parents' experiences and concerns"


class EvaluationResult(BaseModel):
    """Represents evaluation result for a theme."""
    theme_name: str
    coverage_score: int  # 1-5 scale
    coverage_feedback: str
    actionability_score: int  # 1-5 scale
    actionability_feedback: str
    distinctiveness_score: int  # 1-5 scale
    distinctiveness_feedback: str
    relevance_score: int  # 1-5 scale
    relevance_feedback: str
    overall_score: float
    needs_refinement: bool
    refinement_suggestions: List[str]


class OverallEvaluation(BaseModel):
    """Overall evaluation of all themes."""
    theme_evaluations: List[EvaluationResult]
    average_score: float
    is_acceptable: bool
    global_feedback: str


class EvaluationAgent:
    """
    Evaluation Agent that assesses themes based on four criteria:
    1. Coverage: Comprehensively captures important patterns
    2. Actionability: Encapsulates single, clear concept
    3. Distinctiveness: Clearly distinct from other themes
    4. Relevance: Accurately reflects the data (parent experiences)

    Provides feedback until affirmative answer is received.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", expert_criteria: Dict[str, str] = None):
        """
        Initialize the Evaluation Agent.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
            expert_criteria: Optional custom evaluation criteria from cardiac expert
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

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

        Args:
            theme: Theme dictionary to evaluate
            all_themes: List of all themes for distinctiveness check
            original_codes: Original codes for coverage check

        Returns:
            EvaluationResult object
        """
        other_themes = [t for t in all_themes if t["name"] != theme["name"]]
        other_themes_text = "\n".join([f"- {t['name']}: {t['description']}" for t in other_themes])
        codes_text = "\n".join([f"- {code['description']}" for code in original_codes])
        theme_codes_text = "\n".join([f"- {code}" for code in theme.get("codes", [])])

        prompt = f"""You are a clinical research expert evaluating themes from a thematic analysis of parent interviews about their children's congenital heart disease (AAOCA).

Evaluate the following theme based on these four criteria:

1. COVERAGE: {self.criteria.coverage}
2. ACTIONABILITY: {self.criteria.actionability}
3. DISTINCTIVENESS: {self.criteria.distinctiveness}
4. RELEVANCE: {self.criteria.relevance}

THEME TO EVALUATE:
Name: {theme['name']}
Description: {theme['description']}
Associated Codes:
{theme_codes_text}

OTHER THEMES (for distinctiveness comparison):
{other_themes_text}

ORIGINAL CODES (for coverage check):
{codes_text}

For each criterion, provide:
- A score from 1-5 (1=poor, 5=excellent)
- Specific feedback explaining the score
- Suggestions for improvement if score < 4

Also determine:
- Whether this theme needs refinement (true/false)
- Specific refinement suggestions (as a list)

Provide your response as a JSON object with this structure:
{{
  "coverage_score": 1-5,
  "coverage_feedback": "Detailed feedback on coverage",
  "actionability_score": 1-5,
  "actionability_feedback": "Detailed feedback on actionability",
  "distinctiveness_score": 1-5,
  "distinctiveness_feedback": "Detailed feedback on distinctiveness",
  "relevance_score": 1-5,
  "relevance_feedback": "Detailed feedback on relevance",
  "needs_refinement": true/false,
  "refinement_suggestions": ["Suggestion 1", "Suggestion 2", ...]
}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a clinical research expert specializing in qualitative thematic analysis evaluation."},
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
        theme_evaluations = []

        for idx, theme in enumerate(themes):
            print(f"  Evaluating theme {idx + 1}/{len(themes)}: {theme['name']}")
            evaluation = self.evaluate_theme(theme, themes, original_codes)
            theme_evaluations.append(evaluation)

        # Calculate average score
        average_score = sum(e.overall_score for e in theme_evaluations) / len(theme_evaluations)
        is_acceptable = average_score >= acceptance_threshold

        # Generate global feedback
        global_feedback = self._generate_global_feedback(theme_evaluations, average_score, is_acceptable)

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
        is_acceptable: bool
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
        themes_needing_refinement = [e for e in evaluations if e.needs_refinement]

        if is_acceptable:
            feedback = f"ACCEPTABLE: The themes have achieved an average score of {average_score:.2f}/5.0, meeting the acceptance threshold. "
            if themes_needing_refinement:
                feedback += f"However, {len(themes_needing_refinement)} theme(s) could benefit from minor refinements for optimal quality."
        else:
            feedback = f"NEEDS REFINEMENT: The themes have an average score of {average_score:.2f}/5.0, below the acceptance threshold. "
            feedback += f"{len(themes_needing_refinement)} theme(s) require refinement. "

            # Summarize common issues
            coverage_issues = sum(1 for e in evaluations if e.coverage_score < 4)
            actionability_issues = sum(1 for e in evaluations if e.actionability_score < 4)
            distinctiveness_issues = sum(1 for e in evaluations if e.distinctiveness_score < 4)
            relevance_issues = sum(1 for e in evaluations if e.relevance_score < 4)

            issues = []
            if coverage_issues > 0:
                issues.append(f"Coverage ({coverage_issues} themes)")
            if actionability_issues > 0:
                issues.append(f"Actionability ({actionability_issues} themes)")
            if distinctiveness_issues > 0:
                issues.append(f"Distinctiveness ({distinctiveness_issues} themes)")
            if relevance_issues > 0:
                issues.append(f"Relevance ({relevance_issues} themes)")

            if issues:
                feedback += f"Common issues: {', '.join(issues)}."

        return feedback

    def run(
        self,
        themes: List[Dict[str, Any]],
        codes: List[Dict[str, Any]],
        save_path: str = None
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
        overall_eval = self.evaluate_all_themes(themes, codes)

        result = {
            "theme_evaluations": [e.model_dump() for e in overall_eval.theme_evaluations],
            "average_score": overall_eval.average_score,
            "is_acceptable": overall_eval.is_acceptable,
            "global_feedback": overall_eval.global_feedback
        }

        # Save evaluation results if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved evaluation results to {save_path}")

        return result
