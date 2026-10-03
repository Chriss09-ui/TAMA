import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.generation_agent import Chunk, GenerationAgent
from evidence_fixtures import matched_code
from code_mapping import apply_related_codes, build_code_map, render_code_landscape
from code_memos import code_memos_for_model, render_code_memos_markdown
from codebook import exact_merge, render_codebook, semantic_merge
from evidence import link_themes_to_codes, source_document, validate_evidence
from prompts import build_code_extraction_prompt
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

    def test_reflexivity_round_trip_preserves_plain_text_and_markdown(self):
        from tempfile import TemporaryDirectory

        plain = {"position": "我当过实习带教。", "surprise": "表格没人看。", "interest": "",
                 "trouble": "我想替作者下结论。", "noticed": "", "why_noticed": "",
                 "interpretation": "", "how_know": ""}
        note = "第一段\n\n## 自拟标题\n第二段\n## 什么让我感兴趣？\n```text\n代码示例\n```\n最后一段"
        markdown = {"surprise": note, "interest": "独立记录"}
        expected_markdown = {key: "" for key in plain}
        expected_markdown.update(markdown)
        for label, data, expected in (("plain", plain, plain), ("markdown", markdown, expected_markdown)):
            with self.subTest(format=label), TemporaryDirectory() as temp:
                path = Path(temp) / "reflexivity.md"
                save_reflexivity(path, data)
                self.assertEqual(load_reflexivity(path), expected)

    def test_legacy_reflexivity_preserves_custom_headings(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp:
            path = Path(temp) / "reflexivity.md"
            path.write_text("# 研究者自反\n\n## 什么让我惊喜？\n第一段\n## 自拟标题\n第二段\n\n## 什么让我感兴趣？\n另一条记录", encoding="utf-8")
            loaded = load_reflexivity(path)
        self.assertEqual(loaded["surprise"], "第一段\n## 自拟标题\n第二段")
        self.assertEqual(loaded["interest"], "另一条记录")


class CodebookTests(unittest.TestCase):
    def test_exact_merge_transfers_all_members_of_existing_families(self):
        codes = [matched_code(code_id=i, description=f"编码 {i}", excerpt=excerpt, source_start=start)
                 for i, excerpt, start in [(0, "甲", 0), (1, "乙", 1), (2, "甲", 0), (3, "丙", 2)]]
        semantic_merge(codes, [{"code_ids": [0, 1], "name": "统一标签"},
                               {"code_ids": [2, 3], "name": "统一标签"}])
        exact_merge(codes)
        validate_evidence(codes, [source_document("fixture", "甲乙丙")])
        self.assertEqual(codes[0].merged_from, [1, 2, 3])
        self.assertTrue(all(code.merged_into == 0 for code in codes[1:]))
        self.assertTrue(all(not code.merged_from for code in codes[1:]))
        linked = link_themes_to_codes([{"name": "主题", "code_ids": [0]}], codes)[0]
        self.assertEqual(linked["code_ids"], [0, 1, 2, 3])
        exact_merge(codes)
        self.assertEqual(codes[0].merged_from, [1, 2, 3])

    def test_synonymous_codes_merge_and_keep_source_ids(self):
        codes = [
            matched_code(code_id=0, description="每天把抱怨抄进表格", source_chunks=[0], excerpt="抄进表格"),
            matched_code(code_id=1, description="把群消息誊到 Excel", source_chunks=[1], excerpt="誊到 Excel"),
            matched_code(code_id=2, description="第一次进公司", source_chunks=[0], excerpt="第一次进公司"),
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

    def test_invalid_model_merge_preserves_same_labels_at_different_positions(self):
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
        )
        codes = [
            matched_code(code_id=0, description="相同 编码", source_chunks=[0]),
            matched_code(code_id=1, description="相同编码", source_chunks=[1]),
            matched_code(code_id=2, description="另一件事", source_chunks=[1]),
        ]
        agent.consolidate_codebook(codes)
        self.assertIsNone(codes[1].merged_into)
        self.assertIsNone(codes[2].merged_into)
        self.assertIn("无法使用", agent.codebook_note)
        prompt = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn('"code_id": 1', prompt)

    def test_theme_expands_merged_code_ids_and_keeps_rationale(self):
        codes = [
            matched_code(code_id=0, description="抄进表格", source_chunks=[0]),
            matched_code(code_id=1, description="誊到 Excel", source_chunks=[1]),
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


class CodingCycleTests(unittest.TestCase):
    def test_first_cycle_prompt_does_not_ask_for_a_definition(self):
        prompt = build_code_extraction_prompt("一次访谈")
        self.assertIn("不要写定义", prompt)
        self.assertNotIn('"definition"', prompt)
        self.assertNotIn('"include"', prompt)
        self.assertNotIn('"exclude"', prompt)

    def test_extracted_code_drops_a_premature_definition(self):
        agent = GenerationAgent(api_key="test")
        agent.client = Mock()
        agent.client.chat.completions.create.return_value = response({
            "codes": [{
                "name": "有太多东西要了解",
                "description": "有太多东西要了解",
                "definition": "不该在第一轮写死",
                "include": "不该写",
                "exclude": "不该写",
                "in_vivo": True,
                "excerpt": "有太多东西要了解",
            }],
        })
        codes = agent.generate_codes_from_chunk(Chunk(
            chunk_id=0, text="有太多东西要了解", start_word=0, end_word=1,
        ))
        self.assertEqual(codes[0].definition, "")
        self.assertEqual(codes[0].include, "")
        self.assertEqual(codes[0].method, "实境编码")

    def test_code_map_drops_unknown_ids_and_keeps_every_code(self):
        codes = [
            matched_code(code_id=0, description="少", name="少", source_chunks=[0], excerpt="一句"),
            matched_code(code_id=1, description="多", name="多", source_chunks=[1], excerpt="二句"),
            matched_code(code_id=2, description="并入", name="并入", source_chunks=[1], excerpt="三句", merged_into=1),
        ]
        codes[1].merged_from = [2]
        code_map = build_code_map(codes, {
            "categories": [
                {"name": "不存在", "code_ids": [99]},
                {"name": "先出现的一类", "code_ids": [0, 1], "memo": "这两件还不能当成同一件事"},
            ],
            "fewer_categories": [{"name": "更高一层", "code_ids": [0, 1]}],
            "concepts": [{"name": "原话被挪走", "code_ids": [0, 1]}],
        })
        step2 = code_map["iterations"][1]
        listed = [code_id for group in step2["groups"] for code_id in group["code_ids"]]
        self.assertEqual(listed, [0, 1])
        self.assertNotIn(99, listed)
        self.assertIn("无法使用", code_map["note"])
        apply_related_codes(codes, code_map)
        self.assertEqual(codes[0].related_code_ids, [1])
        self.assertEqual(codes[1].related_code_ids, [0])
        landscape = render_code_landscape([code.model_dump() for code in codes], code_map)
        self.assertIn("频次不是重要性", landscape)
        self.assertLess(landscape.find("[0] 少"), landscape.find("[1] 多"))
        self.assertIn("摘录 2 条", landscape)

    def test_code_memo_keeps_the_researcher_note_out_of_the_model_view(self):
        memos = [{
            "date": "2026-09-29",
            "source": "归并",
            "subtitle": "整理客户留言",
            "code_ids": [0, 1],
            "text": "两句都是在搬原话",
            "human_note": "我先不要下结论",
        }]
        markdown = render_code_memos_markdown(memos)
        self.assertIn("我先不要下结论", markdown)
        visible = json.dumps(code_memos_for_model(memos), ensure_ascii=False)
        self.assertIn("两句都是在搬原话", visible)
        self.assertNotIn("我先不要下结论", visible)
        self.assertNotIn("human_note", visible)

    def test_report_names_the_method_and_leaves_the_focus_sheet_blank(self):
        result = {
            "session_name": "cycle",
            "accepted": False,
            "metadata": {"final_average_score": 3},
            "generation": {"analytic_storyline": "记录交上去之后不再被使用"},
            "codes": [
                matched_code(code_id=0, description="少", name="少", excerpt="一句", method="过程编码").model_dump(),
            ],
            "final_themes": [{
                "name": "记录无人再看",
                "description": "交上去的记录很少被使用。",
                "code_ids": [0],
                "counterexample_code_ids": [],
            }],
        }
        result["codes"] = [matched_code(**code).model_dump() for code in result["codes"]]
        lines = "\n".join(text for _kind, text in report_lines(result))
        self.assertIn("过程编码", lines)
        self.assertIn("本轮主题故事线（草稿，模型所写）", lines)
        self.assertIn("聚焦", lines)
        self.assertIn("模型不写、也不读取", lines)
        self.assertIn("（请填写）", lines)
        self.assertIn("频次不是重要性", lines)


class ReportAndPlanTests(unittest.TestCase):
    def test_open_questions_are_grouped_for_current_material_review(self):
        plan = build_next_data_plan(
            codes=[{"code_id": 1, "open_question": "去问产品经理后来有没有人用"}],
            themes=[{
                "name": "对外公开",
                "open_questions": ["核对官网帮助中心的原文"],
                "counterexample_code_ids": [4],
            }],
            memos=[{"theme": "对外公开", "uncertain": "导师和官网哪一边为准"}],
        )
        self.assertIn("当前材料尚无法回答此问题，请核对相关原文与解释边界。", plan["groups"]["原文与来源核对"])
        self.assertIn("核对官网帮助中心的原文", plan["groups"]["原文与来源核对"])
        self.assertIn("导师和官网哪一边为准", plan["groups"]["解释边界"])
        self.assertTrue(any("反例编码 4" in item for item in plan["groups"]["矛盾与反例检查"]))

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
        result["codes"] = [matched_code(**code).model_dump() for code in result["codes"]]
        lines = "\n".join(text for _kind, text in report_lines(result))
        self.assertEqual(lines.count(shared), 1)
        self.assertIn("产品经理直接把记录改成了周报标题", lines)
        self.assertIn("不并进上面的正文", lines)
        self.assertIn("（请在此写下你的诠释。模型不代写这一段。）", lines)
        self.assertIn("边界与局限", lines)
        self.assertIn("所以……", lines)
        self.assertIn("研究者立场", lines)

    def quote_result(self, excerpts, description="短描述", counters=()):
        return {"final_themes": [{"name": "一个模式", "description": description,
                                 "code_ids": list(range(len(excerpts))), "counterexample_code_ids": list(counters)}],
                "codes": [matched_code(code_id=index, description=f"编码{index}", excerpt=excerpt).model_dump()
                          for index, excerpt in enumerate(excerpts)]}

    def test_quote_budget_limits_long_excerpts_and_responds_to_analysis_length(self):
        short = [f"原话{index}" for index in range(6)]
        long = [excerpt + "很长" * 300 for excerpt in short]
        short_quotes = assign_theme_quotes(self.quote_result(short))[0]["supports"]
        long_quotes = assign_theme_quotes(self.quote_result(long))[0]["supports"]
        explained_quotes = assign_theme_quotes(self.quote_result(long, description="分析" * 1000))[0]["supports"]

        self.assertGreater(len(long_quotes), 0)
        self.assertLess(len(long_quotes), len(short_quotes))
        self.assertGreater(len(explained_quotes), len(long_quotes))
        self.assertTrue(all(quote["excerpt"] in long for quote in long_quotes))

    def test_quote_selection_caps_supports_and_counterexamples_per_theme(self):
        excerpts = [f"原话{index}" for index in range(12)]
        pack = assign_theme_quotes(self.quote_result(excerpts, counters=range(6, 12)))[0]
        self.assertEqual(len(pack["supports"]), 4)
        self.assertEqual(len(pack["counters"]), 4)


if __name__ == "__main__":
    unittest.main()
