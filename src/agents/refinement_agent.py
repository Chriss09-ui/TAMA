"""
Refinement Agent for TAMA Framework
Refines themes based on evaluation feedback using four operations:
- Add: Add missing important themes
- Split: Split themes containing multiple concepts
- Combine: Combine repeated or overlapping themes
- Delete: Delete irrelevant themes
"""

from typing import List, Dict, Any
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

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the Refinement Agent.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

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
        themes_text = json.dumps(themes, indent=2)
        evaluations_text = json.dumps(evaluation_results["theme_evaluations"], indent=2)
        codes_text = "\n".join([f"- {code['description']}" for code in codes])

        prompt = f"""You are a qualitative research expert refining themes based on evaluation feedback.

Your task is to create a refinement plan using these four operations:
1. ADD: Add missing important themes that were identified in the evaluation
2. SPLIT: Split themes that contain multiple concepts into separate themes
3. COMBINE: Combine repeated or overlapping themes to eliminate redundancy
4. DELETE: Delete themes that are irrelevant or don't reflect the data

CURRENT THEMES:
{themes_text}

EVALUATION RESULTS:
{evaluations_text}

GLOBAL FEEDBACK:
{evaluation_results["global_feedback"]}

ORIGINAL CODES (for reference):
{codes_text}

INSTRUCTIONS:
- Review the evaluation feedback for each theme
- Identify themes with low scores (<4) in any criterion
- Plan specific refinement operations to address the issues
- Prioritize operations: DELETE first, then COMBINE, then SPLIT, then ADD
- For SPLIT operations, create 2-3 new themes with distinct concepts
- For COMBINE operations, merge themes into a single comprehensive theme
- For ADD operations, identify important missing patterns from codes
- Ensure refined themes will improve coverage, actionability, distinctiveness, and relevance

Provide your response as a JSON object with this structure:
{{
  "operations": [
    {{
      "operation": "add" | "split" | "combine" | "delete",
      "target_themes": ["Theme name(s) to operate on"],
      "rationale": "Why this operation is needed",
      "new_theme": {{
        "name": "New theme name (if applicable)",
        "description": "New theme description (if applicable)",
        "codes": ["Related codes (if applicable)"]
      }}
    }},
    ...
  ],
  "summary": "Brief summary of the refinement plan"
}}

Note:
- For DELETE operations, new_theme should be null
- For SPLIT operations, provide multiple operation entries with different new_theme values
- For COMBINE operations, provide one new_theme that merges the target_themes
- For ADD operations, target_themes can be empty and provide new_theme
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a qualitative research expert specializing in thematic analysis refinement."},
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
