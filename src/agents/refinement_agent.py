"""
Refinement Agent for Threadline Framework
Refines themes based on evaluation feedback using four operations:
- Add: Add missing important themes
- Split: Split themes containing multiple concepts
- Combine: Combine repeated or overlapping themes
- Delete: Delete irrelevant themes
"""

from typing import List, Dict, Any, Optional, Literal
from openai import OpenAI
from pydantic import BaseModel
import json

from prompts import REFINEMENT_SYSTEM_PROMPT, build_refinement_prompt
from research_profile import ResearchProfile
from evidence import link_themes_to_codes, matched_codes


class RefinementOperation(BaseModel):
    """Represents a refinement operation to perform."""
    operation: Literal["add", "split", "combine", "delete"]
    target_themes: List[str]  # Theme names to operate on
    rationale: str
    new_theme: Optional[Dict[str, Any]] = None


class RefinementPlan(BaseModel):
    """Plan of refinement operations to perform."""
    operations: List[RefinementOperation]
    summary: str
    analytic_storyline: str = ""


class RefinementAgent:
    """
    Refinement Agent that refines themes based on evaluation feedback.

    Refinement operations:
    1. ADD: Add missing important themes identified in evaluation
    2. SPLIT: Split themes that contain multiple concepts (low actionability)
    3. COMBINE: Combine repeated or overlapping themes (low distinctiveness)
    4. DELETE: Delete irrelevant themes (low relevance)
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None,
                 study: Optional[ResearchProfile] = None):
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
        self.study = study or ResearchProfile()

    def create_refinement_plan(
        self,
        themes: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        codes: List[Dict[str, Any]],
        memos: Optional[List[Dict[str, Any]]] = None,
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
        prompt = build_refinement_prompt(
            themes=themes,
            theme_evaluations=evaluation_results["theme_evaluations"],
            global_feedback=evaluation_results["global_feedback"],
            codes=[{
                "code_id": code.get("code_id"), "description": code.get("description"),
                "excerpt": str(code.get("excerpt") or "")[:120],
                "context": code.get("context") or "",
                "source": code.get("source") or {},
                "note": code.get("note") or "",
            } for code in matched_codes(codes)],
            study=self.study,
            memos=memos,
        )

        if self.before_model_call:
            self.before_model_call("修订主题")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
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
            summary=result["summary"],
            analytic_storyline=result.get("analytic_storyline") or "",
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
        original_by_name = {theme["name"]: theme for theme in themes}

        for operation in plan.operations:
            if operation.operation in ("add", "split", "combine") and not (
                isinstance(operation.new_theme, dict) and operation.new_theme.get("name")
            ):
                raise ValueError(f"{operation.operation} operation requires a named new_theme")
            if operation.operation == "split" and len(operation.target_themes) != 1:
                raise ValueError("split operation requires one target theme")
            if operation.operation == "combine" and len(operation.target_themes) < 2:
                raise ValueError("combine operation requires at least two target themes")
            if operation.new_theme is not None:
                parents = [original_by_name[name] for name in operation.target_themes if name in original_by_name]
                counters = list(operation.new_theme.get("counterexample_code_ids") or [])
                questions = list(operation.new_theme.get("open_questions") or [])
                for parent in parents:
                    counters.extend(parent.get("counterexample_code_ids") or [])
                    questions.extend(parent.get("open_questions") or [])
                    if parent.get("uncertain"):
                        questions.append(parent["uncertain"])
                operation.new_theme["counterexample_code_ids"] = list(dict.fromkeys(counters))
                operation.new_theme["open_questions"] = list(dict.fromkeys(questions))
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
        save_path: str = None,
        memos: Optional[List[Dict[str, Any]]] = None,
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
        plan = self.create_refinement_plan(themes, evaluation_results, codes, memos=memos)

        print(f"\nRefinement Plan Summary: {plan.summary}")
        print(f"Total operations: {len(plan.operations)}")

        print("\nApplying refinement operations...")
        refined_themes = link_themes_to_codes(self.apply_refinement_plan(themes, plan), codes)

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
        result["analytic_storyline"] = plan.analytic_storyline

        # Save refinement results if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved refinement results to {save_path}")

        return result
