"""
Refinement Agent for TAMA Framework
Refines themes based on evaluation feedback using four operations:
- Add: Add missing important themes
- Split: Split themes containing multiple concepts
- Combine: Combine repeated or overlapping themes
- Delete: Delete irrelevant themes
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel
import json


class RefinementOperation(BaseModel):
    """Represents a refinement operation to perform."""
    operation: str  # "add", "split", "combine", "delete"
    target_themes: List[str]  # Theme names to operate on
    rationale: str
    new_theme: Dict[str, Any] = None  # For add/split operations


class RefinementPlan(BaseModel):
    """Plan of refinement operations to perform."""
    operations: List[RefinementOperation]
    summary: str


class RefinementAgent:
    """
    Refinement Agent that refines themes based on evaluation feedback.

    Refinement operations:
    1. ADD: Add missing important themes identified in evaluation
    2. SPLIT: Split themes that contain multiple concepts (low actionability)
    3. COMBINE: Combine repeated or overlapping themes (low distinctiveness)
    4. DELETE: Delete irrelevant themes (low relevance)
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None):
        """
        Initialize the Refinement Agent.

        Args:
            api_key: API key for the selected model provider
            model: Model to use (default: gpt-4o)
            base_url: Optional OpenAI-compatible API endpoint
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.before_model_call = None

    def create_refinement_plan(
        self,
        themes: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        codes: List[Dict[str, Any]]
    ) -> RefinementPlan:
        """
        Create a plan for refining themes based on evaluation feedback.

        Args:
            themes: Current list of themes
            evaluation_results: Results from evaluation agent
            codes: Original codes for reference

        Returns:
            RefinementPlan object
        """
        themes_text = json.dumps(themes, ensure_ascii=False, indent=2)
        evaluations_text = json.dumps(evaluation_results["theme_evaluations"], ensure_ascii=False, indent=2)
        codes_text = "\n".join([f"- {code['description']}" for code in codes])

        prompt = f"""你是一名质性研究者，正在根据评估反馈修订主题。

请使用以下四种操作制定修订计划；"operation" 字段使用括号中的英文值：
1. 增加（"add"）：补充评估中发现缺失的重要主题
2. 拆分（"split"）：将包含多个概念的主题拆成不同主题
3. 合并（"combine"）：合并重复或重叠的主题
4. 删除（"delete"）：删除与材料无关或不能反映材料的主题

当前主题：
{themes_text}

评估结果：
{evaluations_text}

总体反馈：
{evaluation_results["global_feedback"]}

原始编码（供参考）：
{codes_text}

要求：
- 查看每个主题的评估反馈，找出任一标准低于 4 分的主题
- 针对问题制定具体操作，优先顺序为删除、合并、拆分、增加
- 拆分时创建 2 至 3 个概念不同的新主题；合并时创建一个涵盖相关概念的新主题
- 增加主题时，从原始编码中识别遗漏的重要规律
- 修订应改善覆盖度、概念清晰度、区分度和相关性
- 所有新增或修改的主题都必须有提供的编码作为依据，不编造背景或证据
- 新主题、操作理由及计划摘要使用与当前主题和编码相同的语言

只返回符合以下结构的 JSON 对象，不添加解释或 Markdown；保留英文键名及操作值：
{{
  "operations": [
    {{
      "operation": "add",
      "target_themes": [],
      "rationale": "需要执行该操作的原因",
      "new_theme": {{
        "name": "新主题名称",
        "description": "新主题描述",
        "codes": ["相关的原始编码描述"]
      }}
    }}
  ],
  "summary": "修订计划的简短摘要"
}}

补充说明：
- 无需修订时，"operations" 返回空数组
- 删除操作的 "new_theme" 为 null
- 拆分操作需要返回多条记录，各自提供不同的 "new_theme"
- 合并操作提供一个合并后的 "new_theme"
- 增加操作的 "target_themes" 可以为空数组，并提供 "new_theme"
"""

        if self.before_model_call:
            self.before_model_call("修订主题")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一名质性研究者。仅依据提供的编码和评估反馈修订主题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        operations = []
        for op_data in result["operations"]:
            operations.append(RefinementOperation(
                operation=op_data["operation"],
                target_themes=op_data["target_themes"],
                rationale=op_data["rationale"],
                new_theme=op_data.get("new_theme")
            ))

        return RefinementPlan(
            operations=operations,
            summary=result["summary"]
        )

    def apply_refinement_plan(
        self,
        themes: List[Dict[str, Any]],
        plan: RefinementPlan
    ) -> List[Dict[str, Any]]:
        """
        Apply the refinement plan to produce refined themes.

        Args:
            themes: Current list of themes
            plan: RefinementPlan to apply

        Returns:
            Refined list of themes
        """
        refined_themes = themes.copy()

        for operation in plan.operations:
            if operation.operation == "delete":
                refined_themes = self._apply_delete(refined_themes, operation)
            elif operation.operation == "combine":
                refined_themes = self._apply_combine(refined_themes, operation)
            elif operation.operation == "split":
                refined_themes = self._apply_split(refined_themes, operation)
            elif operation.operation == "add":
                refined_themes = self._apply_add(refined_themes, operation)

        return refined_themes

    def _apply_delete(
        self,
        themes: List[Dict[str, Any]],
        operation: RefinementOperation
    ) -> List[Dict[str, Any]]:
        """Delete themes specified in operation."""
        print(f"  DELETE: Removing {len(operation.target_themes)} theme(s)")
        print(f"    Rationale: {operation.rationale}")

        return [t for t in themes if t["name"] not in operation.target_themes]

    def _apply_combine(
        self,
        themes: List[Dict[str, Any]],
        operation: RefinementOperation
    ) -> List[Dict[str, Any]]:
        """Combine themes into a single theme."""
        print(f"  COMBINE: Merging {len(operation.target_themes)} theme(s)")
        print(f"    Rationale: {operation.rationale}")
        print(f"    New theme: {operation.new_theme['name']}")

        # Remove target themes and add combined theme
        filtered_themes = [t for t in themes if t["name"] not in operation.target_themes]
        filtered_themes.append(operation.new_theme)

        return filtered_themes

    def _apply_split(
        self,
        themes: List[Dict[str, Any]],
        operation: RefinementOperation
    ) -> List[Dict[str, Any]]:
        """Split a theme into multiple themes."""
        print(f"  SPLIT: Splitting theme '{operation.target_themes[0]}'")
        print(f"    Rationale: {operation.rationale}")
        print(f"    New theme: {operation.new_theme['name']}")

        # Note: Multiple SPLIT operations for same theme will be in sequence
        # Only remove the target theme once if it exists
        result_themes = []
        theme_removed = False

        for theme in themes:
            if theme["name"] in operation.target_themes and not theme_removed:
                theme_removed = True
                # Skip this theme, it will be replaced by new split themes
                continue
            result_themes.append(theme)

        # Add the new split theme
        result_themes.append(operation.new_theme)

        return result_themes

    def _apply_add(
        self,
        themes: List[Dict[str, Any]],
        operation: RefinementOperation
    ) -> List[Dict[str, Any]]:
        """Add a new theme."""
        print(f"  ADD: Adding new theme '{operation.new_theme['name']}'")
        print(f"    Rationale: {operation.rationale}")

        themes.append(operation.new_theme)
        return themes

    def run(
        self,
        themes: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        codes: List[Dict[str, Any]],
        save_path: str = None
    ) -> Dict[str, Any]:
        """
        Run the refinement process on themes based on evaluation feedback.

        Args:
            themes: Current list of themes
            evaluation_results: Results from evaluation agent
            codes: Original codes for reference
            save_path: Optional path to save refinement results

        Returns:
            Dictionary containing refinement plan and refined themes
        """
        print("\nCreating refinement plan...")
        plan = self.create_refinement_plan(themes, evaluation_results, codes)

        print(f"\nRefinement Plan Summary: {plan.summary}")
        print(f"Total operations: {len(plan.operations)}")

        print("\nApplying refinement operations...")
        refined_themes = self.apply_refinement_plan(themes, plan)

        print(f"\nRefinement complete:")
        print(f"  Original themes: {len(themes)}")
        print(f"  Refined themes: {len(refined_themes)}")

        result = {
            "original_themes": themes,
            "refinement_plan": {
                "operations": [op.model_dump() for op in plan.operations],
                "summary": plan.summary
            },
            "refined_themes": refined_themes,
            "theme_count_before": len(themes),
            "theme_count_after": len(refined_themes)
        }

        # Save refinement results if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved refinement results to {save_path}")

        return result
