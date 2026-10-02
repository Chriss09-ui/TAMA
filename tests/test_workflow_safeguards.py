import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.generation_agent import Chunk, Code, GenerationAgent
from evidence_fixtures import matched_code
from agents.refinement_agent import RefinementAgent, RefinementOperation, RefinementPlan
from prompts import build_code_extraction_prompt
from research_profile import resolve_profile
from tama import TAMAFramework


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])


def generation_data():
    return {
        "chunks": ["fragment"],
        "codes": [matched_code(code_id=0, description="编码", excerpt="private quote").model_dump()],
        "themes": [{"name": "模式", "description": "描述", "code_ids": [0]}],
        "analytic_storyline": "一个初始分析故事线",
    }


def evaluation_data(score=3.0, accepted=False):
    return {
        "theme_evaluations": [], "average_score": score,
        "is_acceptable": accepted, "global_feedback": "需继续审查",
    }


class WorkflowSafeguardTests(unittest.TestCase):
    def test_generic_prompt_has_no_enterprise_assumption_and_focus_is_injected(self):
        generic = build_code_extraction_prompt("一次访谈")
        self.assertNotIn("企业具体能力", generic)
        self.assertNotIn("内部证据", generic)
        study = resolve_profile("enterprise_evidence", "能力从哪里来？", ["研发", "营销"])
        focused = build_code_extraction_prompt("一次访谈", study)
        self.assertIn("能力从哪里来？", focused)
        self.assertIn("研发、营销", focused)

    def test_enterprise_profile_uses_study_rubric_with_custom_override(self):
        with patch("tama.GenerationAgent"), patch("tama.EvaluationAgent") as evaluation, \
                patch("tama.RefinementAgent"):
            framework = TAMAFramework(
                api_key="test", profile="enterprise_evidence",
                expert_criteria={"coverage": "自定义覆盖要求"},
            )
        criteria = evaluation.call_args.kwargs["expert_criteria"]
        self.assertEqual(criteria["coverage"], "自定义覆盖要求")
        self.assertIn("候选机制", criteria["relevance"])
        self.assertEqual(framework.expert_criteria, criteria)

    def test_long_code_list_uses_bounded_batches_and_final_consolidation(self):
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        agent.client.chat.completions.create.side_effect = [
            response({"themes": [{"name": "A", "description": "A", "code_ids": [0]}]}),
            response({"themes": [{"name": "B", "description": "B", "code_ids": [40]}]}),
            response({"themes": [{"name": "C", "description": "C", "code_ids": [80]}]}),
            response({"analytic_storyline": "A、B、C 形成整体", "themes": [
                {"name": "整体模式", "description": "整体描述", "code_ids": [0, 40, 80]},
            ]}),
        ]
        codes = [matched_code(code_id=i, description=f"编码{i}", source_chunks=[i // 40]) for i in range(85)]
        themes = agent.generate_themes(codes)
        self.assertEqual(len(agent.client.chat.completions.create.call_args_list), 4)
        self.assertEqual(themes[0].code_ids, [0, 40, 80])
        self.assertEqual(agent.analytic_storyline, "A、B、C 形成整体")
        first_prompt = agent.client.chat.completions.create.call_args_list[0].kwargs["messages"][1]["content"]
        self.assertNotIn('"code_id": 80', first_prompt)

    def test_same_labels_reach_the_model_separately_without_implicit_theme_links(self):
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        agent.client.chat.completions.create.return_value = response({
            "themes": [{"name": "模式", "description": "描述", "code_ids": [0]}],
        })
        themes = agent.generate_themes([
            matched_code(code_id=0, description="相同 编码", source_chunks=[0]),
            matched_code(code_id=1, description="相同编码", source_chunks=[1]),
        ])
        self.assertEqual(themes[0].code_ids, [0])
        prompt = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn('"code_id": 0', prompt)
        self.assertIn('"code_id": 1', prompt)

    def test_malformed_model_json_fails_explicitly(self):
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
        )
        chunk = Chunk(chunk_id=0, text="受访者：一次经历。", start_word=0, end_word=1)
        with self.assertRaises(json.JSONDecodeError):
            agent.generate_codes_from_chunk(chunk)

    def test_in_memory_run_creates_no_directory_and_saved_result_keeps_excerpt(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch("tama.GenerationAgent") as generation, \
                patch("tama.EvaluationAgent") as evaluation, \
                patch("tama.RefinementAgent"):
            generation.return_value.run.return_value = generation_data()
            evaluation.return_value.run.return_value = evaluation_data(4.0, True)
            output = Path(temp) / "outputs"
            framework = TAMAFramework(api_key="test", output_dir=str(output), corpus_review=False)
            memory = framework.run_analysis("private quote", save_final=False, source_metadata={"source_id": "fixture"})
            self.assertFalse(output.exists())
            self.assertEqual(memory["codes"][0]["excerpt"], "private quote")
            self.assertEqual(evaluation.return_value.run.call_args.kwargs["acceptance_threshold"], 4.0)

            saved = framework.run_analysis(
                "private quote", session_name="study", source_metadata={"source_id": "fixture"}, save_final=True,
                save_intermediate=False,
            )
            persisted = (output / "study" / "00_final_results.json").read_text(encoding="utf-8")
            self.assertIn("private quote", persisted)
            codebook = (output / "study" / "05_codebook.md").read_text(encoding="utf-8")
            self.assertIn("private quote", codebook)
            self.assertIn("编码簿", codebook)
            self.assertTrue((output / "study" / "04_memos.md").is_file())
            self.assertTrue((output / "study" / "next_data_plan.md").is_file())
            self.assertEqual(saved["codes"][0]["excerpt"], "private quote")
            again = framework.run_analysis(
                "private quote", session_name="study", source_metadata={"source_id": "fixture"}, save_final=True, save_intermediate=False,
            )
            self.assertNotEqual(again["session_name"], "study")
            self.assertTrue((output / again["session_name"] / "00_final_results.json").exists())

    def test_stagnant_scores_stop_without_extra_refinement(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch("tama.GenerationAgent") as generation, \
                patch("tama.EvaluationAgent") as evaluation, \
                patch("tama.RefinementAgent") as refinement:
            generation.return_value.run.return_value = generation_data()
            evaluation.return_value.run.side_effect = [evaluation_data(3.0)] * 3
            refinement.return_value.run.return_value = {
                "refined_themes": generation_data()["themes"],
                "refinement_plan": {"summary": "调整"},
                "theme_count_before": 1, "theme_count_after": 1,
            }
            framework = TAMAFramework(api_key="test", output_dir=temp, max_iterations=6, corpus_review=False)
            result = framework.run_analysis("private quote", save_final=False, save_intermediate=False, source_metadata={"source_id": "fixture"})
            self.assertEqual(result["stop_reason"], "no_improvement")
            self.assertEqual([item["average_score"] for item in result["score_history"]], [3.0] * 3)
            self.assertEqual(refinement.return_value.run.call_count, 2)

    def test_refinement_four_operations_and_run_json_error(self):
        agent = RefinementAgent(api_key="test")
        original = [
            {"name": "删除项"}, {"name": "合并甲"}, {"name": "合并乙"}, {"name": "拆分项"},
        ]
        operations = [
            RefinementOperation(operation="delete", target_themes=["删除项"], rationale="背景"),
            RefinementOperation(operation="combine", target_themes=["合并甲", "合并乙"], rationale="重复", new_theme={"name": "合并后"}),
            RefinementOperation(operation="split", target_themes=["拆分项"], rationale="过宽", new_theme={"name": "拆分一"}),
            RefinementOperation(operation="split", target_themes=["拆分项"], rationale="过宽", new_theme={"name": "拆分二"}),
            RefinementOperation(operation="add", target_themes=[], rationale="遗漏", new_theme={"name": "新增"}),
        ]
        result = agent.apply_refinement_plan(original, RefinementPlan(operations=operations, summary="完成"))
        self.assertEqual([theme["name"] for theme in result], ["合并后", "拆分一", "拆分二", "新增"])
        self.assertEqual(len(original), 4)

        agent.client = Mock()
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{bad json"))]
        )
        with self.assertRaises(json.JSONDecodeError):
            agent.run(original, {"theme_evaluations": [], "global_feedback": ""}, [], save_path=None)

        agent.client.chat.completions.create.return_value = response({
            "operations": [{
                "operation": "add", "target_themes": [], "rationale": "遗漏模式",
                "new_theme": {"name": "补充主题", "description": "描述", "code_ids": [0]},
            }],
            "summary": "补充一个主题",
        })
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "refinement.json"
            saved = agent.run(
                [{"name": "已有主题"}],
                {"theme_evaluations": [], "global_feedback": "遗漏"},
                [{"code_id": 0, "description": "编码"}], save_path=str(path),
            )
            self.assertEqual(saved["theme_count_before"], 1)
            self.assertEqual(saved["theme_count_after"], 2)
            self.assertEqual(json.loads(path.read_text())["refined_themes"][1]["name"], "补充主题")


if __name__ == "__main__":
    unittest.main()
