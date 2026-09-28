import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.generation_agent import Code, GenerationAgent
from codebook import exact_merge, render_codebook, semantic_merge
from memos import (
    HUMAN_OPEN,
    apply_human_notes,
    build_theme_memos,
    human_notes_from_markdown,
    memos_for_model,
    render_memos_markdown,
)
from next_data_plan import build_next_data_plan
from prompts import build_full_evaluation_prompt, build_refinement_prompt
from reflexivity import load_reflexivity, save_reflexivity
from reporting import assign_theme_quotes, report_lines


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])


class ScaleAndMemoTests(unittest.TestCase):
    def test_pattern_scale_marks_background_sections_as_low(self):
        prompt = build_full_evaluation_prompt(
            criteria=SimpleNamespace(
                coverage="覆盖", actionability="模式", distinctiveness="区分", relevance="有据",
            ),
            theme={"name": "实习背景", "description": "专业介绍"},
            other_themes=[],
            original_codes=[],
        )
        self.assertIn("模式性只能是 1 分或 2 分", prompt)
        self.assertIn("话题罗列或背景性章节", prompt)
        self.assertNotIn("概念清晰度", prompt)
        self.assertNotIn("较差", prompt)

    def test_human_memo_is_kept_in_markdown_and_kept_out_of_the_prompt(self):
        themes = [{
            "name": "记录无人再看",
            "description": "交上去的记录很少被使用。",
            "code_ids": [0, 1],
            "counterexample_code_ids": [2],
            "rationale": "三句都是把客户原话搬进表格",
            "uncertain": "不知道后来有没有人打开",
        }]
        memos = build_theme_memos(themes, iteration=0)
        self.assertEqual(themes[0]["counterexample_code_ids"], [2])
        rewritten = apply_human_notes(memos, {"记录无人再看": "我自己当过带教，这里先别下结论。"})
        markdown = render_memos_markdown(rewritten)
        self.assertIn(HUMAN_OPEN, markdown)
        self.assertIn("我自己当过带教", markdown)
        self.assertEqual(
            human_notes_from_markdown(markdown)["记录无人再看"],
            "我自己当过带教，这里先别下结论。",
        )
        prompt = build_refinement_prompt(
            themes=themes,
            theme_evaluations=[],
            global_feedback="",
            codes=[],
            memos=rewritten,
        )
        self.assertIn("三句都是把客户原话搬进表格", prompt)
        self.assertNotIn("我自己当过带教", prompt)
        self.assertNotIn("human_note", json.dumps(memos_for_model(rewritten), ensure_ascii=False))

    def test_reflexivity_file_round_trip(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp:
            path = Path(temp) / "reflexivity.md"
            save_reflexivity(path, {
                "position": "我当过实习带教。",
                "surprise": "表格没人看。",
                "interest": "",
                "trouble": "我想替作者下结论。",
                "noticed": "",
                "why_noticed": "",
                "interpretation": "",
                "how_know": "",
            })
            loaded = load_reflexivity(path)
        self.assertEqual(loaded["position"], "我当过实习带教。")
        self.assertEqual(loaded["trouble"], "我想替作者下结论。")
        self.assertEqual(loaded["interest"], "")


class CodebookTests(unittest.TestCase):
    def test_synonymous_codes_merge_and_keep_source_ids(self):
        codes = [
            Code(code_id=0, description="每天把抱怨抄进表格", source_chunks=[0], excerpt="抄进表格"),
            Code(code_id=1, description="把群消息誊到 Excel", source_chunks=[1], excerpt="誊到 Excel"),
            Code(code_id=2, description="第一次进公司", source_chunks=[0], excerpt="第一次进公司"),
        ]
        semantic_merge(codes, [{
            "name": "整理客户留言",
            "definition": "把客户原话搬进表格",
            "include": "抄录、誊写、整理留言",
            "exclude": "阅读或改写这些记录",
            "example_code_id": 0,
            "code_ids": [0, 1],
        }])
        self.assertEqual(codes[0].merged_from, [1])
        self.assertEqual(codes[1].merged_into, 0)
        self.assertEqual(codes[0].definition, "把客户原话搬进表格")
        self.assertIsNone(codes[2].merged_into)
        text = render_codebook([code.model_dump() for code in codes])
        self.assertIn("把客户原话搬进表格", text)
        self.assertIn("誊到 Excel", text)
        self.assertIn("第一次进公司", text)

    def test_invalid_model_merge_keeps_exact_duplicates_only(self):
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
        )
        codes = [
            Code(code_id=0, description="相同 编码", source_chunks=[0]),
            Code(code_id=1, description="相同编码", source_chunks=[1]),
            Code(code_id=2, description="另一件事", source_chunks=[1]),
        ]
        agent.consolidate_codebook(codes)
        self.assertEqual(codes[1].merged_into, 0)
        self.assertIsNone(codes[2].merged_into)
        self.assertIn("无法使用", agent.codebook_note)
        prompt = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertNotIn('"code_id": 1', prompt)

    def test_theme_expands_merged_code_ids_and_keeps_rationale(self):
        codes = [
            Code(code_id=0, description="抄进表格", source_chunks=[0]),
            Code(code_id=1, description="誊到 Excel", source_chunks=[1]),
        ]
        exact_merge(codes)
        semantic_merge(codes, [{
            "name": "整理客户留言",
            "definition": "把客户原话搬进表格",
            "include": "抄录",
            "exclude": "改写",
            "code_ids": [0, 1],
        }])
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        agent.client.chat.completions.create.return_value = response({
            "analytic_storyline": "记录交上去之后不再被使用",
            "themes": [{
                "name": "记录无人再看",
                "description": "交上去的记录很少被使用。",
                "code_ids": [0],
                "counterexample_code_ids": [],
                "rationale": "两句都是把客户原话搬进表格",
                "uncertain": "不知道后来有没有人打开",
            }],
        })
        themes = agent.generate_themes(codes)
        self.assertEqual(themes[0].code_ids, [0, 1])
        self.assertEqual(themes[0].rationale, "两句都是把客户原话搬进表格")
        self.assertIn("整理客户留言", agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"])


class ReportAndPlanTests(unittest.TestCase):
    def test_open_questions_are_grouped_for_the_next_round(self):
        plan = build_next_data_plan(
            codes=[{"code_id": 1, "open_question": "去问产品经理后来有没有人用"}],
            themes=[{
                "name": "对外公开",
                "open_questions": ["核对官网帮助中心的原文"],
                "counterexample_code_ids": [4],
            }],
            memos=[{"theme": "对外公开", "uncertain": "导师和官网哪一边为准"}],
        )
        self.assertIn("去问产品经理后来有没有人用", plan["groups"]["需补访角色"])
        self.assertIn("核对官网帮助中心的原文", plan["groups"]["需查材料"])
        self.assertIn("导师和官网哪一边为准", plan["groups"]["需查材料"])
        self.assertTrue(any("反例编码 4" in item for item in plan["groups"]["需核对信息"]))

    def test_report_keeps_one_copy_of_a_quote_and_leaves_blanks(self):
        shared = "表格交上去之后几乎没有人再打开"
        result = {
            "session_name": "blog",
            "accepted": False,
            "metadata": {"final_average_score": 3.2},
            "refinement_iterations": 1,
            "codes": [
                {"code_id": 0, "description": "抄进表格", "excerpt": shared, "statement_type": "participant_report"},
                {"code_id": 1, "description": "誊到 Excel", "excerpt": "把群消息誊到 Excel", "statement_type": "participant_report"},
                {"code_id": 2, "description": "改成周报标题", "excerpt": "产品经理直接把记录改成了周报标题", "statement_type": "participant_report"},
                {"code_id": 3, "description": "同样的原话", "excerpt": shared, "statement_type": "participant_report"},
            ],
            "final_themes": [
                {
                    "name": "记录无人再看",
                    "description": "交上去的记录很少被使用。",
                    "rationale": "这些句子都是把客户原话搬进表格。",
                    "code_ids": [0, 1],
                    "counterexample_code_ids": [2],
                    "open_questions": ["后来有没有人打开表格"],
                },
                {
                    "name": "另一个主题",
                    "description": "重复使用了同一句。",
                    "code_ids": [3],
                    "counterexample_code_ids": [],
                },
            ],
            "memos": [],
        }
        lines = "\n".join(text for _kind, text in report_lines(result))
        self.assertEqual(lines.count(shared), 1)
        self.assertIn("产品经理直接把记录改成了周报标题", lines)
        self.assertIn("不并进上面的正文", lines)
        self.assertIn("（请在此写下你的诠释。模型不代写这一段。）", lines)
        self.assertIn("边界与局限", lines)
        self.assertIn("所以……", lines)
        self.assertIn("研究者立场", lines)

    def test_long_quotes_do_not_fill_the_theme_section(self):
        excerpts = [f"原话{index}" + ("很长" * 300) for index in range(6)]
        result = {
            "session_name": "long",
            "accepted": True,
            "metadata": {"final_average_score": 4},
            "final_themes": [{
                "name": "一个模式",
                "description": "短描述",
                "code_ids": list(range(6)),
                "counterexample_code_ids": [],
            }],
            "codes": [
                {"code_id": index, "description": f"编码{index}", "excerpt": excerpt}
                for index, excerpt in enumerate(excerpts)
            ],
        }
        packs = assign_theme_quotes(result)
        self.assertLess(len(packs[0]["supports"]), 6)
        self.assertLessEqual(len(packs[0]["supports"]), 4)


if __name__ == "__main__":
    unittest.main()
