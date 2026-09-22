import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tama import TAMAFramework


class FrameworkResultTests(unittest.TestCase):
    def test_accepted_run_keeps_final_evaluation_audit_data(self):
        generation_result = {
            "chunks": ["访谈文本"],
            "codes": [{"description": "编码"}],
            "themes": [{"name": "主题", "description": "描述", "codes": ["编码"]}],
            "chunking": {
                "strategy": "balanced", "total_characters": 4,
                "estimated_tokens": 3, "target_size": 3,
                "hard_limit": 2400, "planned_chunks": 1, "num_chunks": 1,
            },
        }
        evaluation_result = {
            "theme_evaluations": [{"theme_name": "主题", "flagged_for_review": True}],
            "average_score": 4.0,
            "is_acceptable": True,
            "global_feedback": "已达标，建议人工复核。",
            "flagged_themes": ["主题"],
        }

        with tempfile.TemporaryDirectory() as output_dir, \
                patch("tama.GenerationAgent") as generation_cls, \
                patch("tama.EvaluationAgent") as evaluation_cls, \
                patch("tama.RefinementAgent") as refinement_cls:
            generation_cls.return_value.run.return_value = generation_result
            evaluation_cls.return_value.run.return_value = evaluation_result
            framework = TAMAFramework(api_key="test", output_dir=output_dir)

            result = framework.run_analysis(
                "访谈文本", session_name="audit-test", save_intermediate=False,
            )

            self.assertEqual(result["final_evaluation"], evaluation_result)
            self.assertEqual(result["configuration"]["chunk_strategy"], "balanced")
            self.assertEqual(result["generation"]["chunking"], generation_result["chunking"])
            refinement_cls.return_value.run.assert_not_called()
            saved_path = Path(output_dir) / "audit-test" / "00_final_results.json"
            with saved_path.open(encoding="utf-8") as saved_file:
                self.assertEqual(json.load(saved_file)["final_evaluation"], evaluation_result)


if __name__ == "__main__":
    unittest.main()
