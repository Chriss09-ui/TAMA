"""
Evaluation Agent for TAMA Framework
Evaluates generated themes based on four criteria: Coverage, Actionability, Distinctiveness, and Relevance.
Provides feedback for refinement until affirmative answer is received.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel
import json


class EvaluationCriteria(BaseModel):
    """Represents evaluation criteria for themes."""
    coverage: str = "应覆盖所提供编码中的重要规律"
    actionability: str = "应表达一个清楚、便于理解的概念"
    distinctiveness: str = "应与其他主题有明确区分，避免重叠"
    relevance: str = "应有提供的编码作为依据，并准确反映受访者表达的意思"


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
    4. Relevance: Accurately reflects the supplied data

    Provides feedback until affirmative answer is received.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        expert_criteria: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the Evaluation Agent.

        Args:
            api_key: API key for the selected model provider
            model: Model to use (default: gpt-4o)
            expert_criteria: Optional study-specific evaluation criteria from a researcher
            base_url: Optional OpenAI-compatible API endpoint
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.before_model_call = None

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

        prompt = f"""你是一名质性研究者，正在评估访谈分析形成的主题。
仅使用提供的主题和编码，不预设材料未说明的人群或研究主题。

请依据以下四项标准评估主题：

1. 覆盖度：{self.criteria.coverage}
2. 概念清晰度：{self.criteria.actionability}
3. 区分度：{self.criteria.distinctiveness}
4. 相关性：{self.criteria.relevance}

待评估主题：
名称：{theme['name']}
描述：{theme['description']}
关联编码：
{theme_codes_text}

其他主题（用于比较区分度）：
{other_themes_text}

原始编码（用于检查覆盖度）：
{codes_text}

每项标准均需提供：
- 1 到 5 分的整数评分（1 分较差，5 分优秀）
- 解释评分依据的具体反馈
- 低于 4 分时给出改进建议
- 如果编码不足以支持某项判断，指出证据缺口，不编造依据
- 反馈使用与主题和编码相同的语言

同时判断：
- 该主题是否需要修订（布尔值）
- 具体的修订建议（列表）

只返回符合以下结构的 JSON 对象，不添加解释或 Markdown；保留英文键名：
{{
  "coverage_score": 4,
  "coverage_feedback": "覆盖度的具体反馈",
  "actionability_score": 4,
  "actionability_feedback": "概念清晰度的具体反馈",
  "distinctiveness_score": 4,
  "distinctiveness_feedback": "区分度的具体反馈",
  "relevance_score": 4,
  "relevance_feedback": "相关性的具体反馈",
  "needs_refinement": false,
  "refinement_suggestions": []
}}
"""

        if self.before_model_call:
            self.before_model_call(f"评估主题 · {theme['name']}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一名质性研究者。仅依据提供的材料评估主题。"},
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
            feedback = f"已达标：主题平均分为 {average_score:.2f}/5.0，达到验收标准。"
            if themes_needing_refinement:
                feedback += f"仍有 {len(themes_needing_refinement)} 个主题可进一步完善。"
        else:
            feedback = f"需要修订：主题平均分为 {average_score:.2f}/5.0，低于验收标准。"
            feedback += f"其中 {len(themes_needing_refinement)} 个主题需要修订。"

            # Summarize common issues
            coverage_issues = sum(1 for e in evaluations if e.coverage_score < 4)
            actionability_issues = sum(1 for e in evaluations if e.actionability_score < 4)
            distinctiveness_issues = sum(1 for e in evaluations if e.distinctiveness_score < 4)
            relevance_issues = sum(1 for e in evaluations if e.relevance_score < 4)

            issues = []
            if coverage_issues > 0:
                issues.append(f"覆盖度（{coverage_issues} 个主题）")
            if actionability_issues > 0:
                issues.append(f"概念清晰度（{actionability_issues} 个主题）")
            if distinctiveness_issues > 0:
                issues.append(f"区分度（{distinctiveness_issues} 个主题）")
            if relevance_issues > 0:
                issues.append(f"相关性（{relevance_issues} 个主题）")

            if issues:
                feedback += f"常见问题：{'、'.join(issues)}。"

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
