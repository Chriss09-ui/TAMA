"""All prompts and evaluation rubrics sent to language or decision models."""

import json
from typing import Any, Dict, Iterable, Mapping, Sequence


CODE_EXTRACTION_SYSTEM_PROMPT = (
    "你是一名质性研究者。所有编码都必须有提供的访谈文本作为依据。"
)
THEME_GENERATION_SYSTEM_PROMPT = (
    "你是一名质性研究者。所有主题都必须有提供的编码作为依据。"
)
EVALUATION_FEEDBACK_SYSTEM_PROMPT = (
    "你是一名质性研究者。仅依据提供的材料和评分撰写评估反馈。"
)
FULL_EVALUATION_SYSTEM_PROMPT = (
    "你是一名质性研究者。仅依据提供的材料评估主题。"
)
REFINEMENT_SYSTEM_PROMPT = (
    "你是一名质性研究者。仅依据提供的编码和评估反馈修订主题。"
)
LLM_DECISION_SYSTEM_PROMPT = (
    "你是一名质性研究者。仅依据提供的材料回答问题。"
)

DEFAULT_COVERAGE_CRITERION = "应覆盖所提供编码中的重要规律"
DEFAULT_ACTIONABILITY_CRITERION = "应表达一个清楚、便于理解的概念"
DEFAULT_DISTINCTIVENESS_CRITERION = "应与其他主题有明确区分，避免重叠"
DEFAULT_RELEVANCE_CRITERION = "应有提供的编码作为依据，并准确反映受访者表达的意思"

EVALUATION_SCALE = (
    "1 分：较差",
    "2 分：较弱",
    "3 分：一般",
    "4 分：良好",
    "5 分：优秀",
)

API_CONNECTION_TEST_PROMPT = "请回复 OK。"
JEV_CONNECTION_TEST_STATE = "连接测试"
JEV_CONNECTION_TEST_INSTRUCTIONS = "这是一次连接测试。请回答：1+1 是否等于 2？"


def build_code_extraction_prompt(chunk_text: str) -> str:
    return f"""你是一名质性研究者，正在对访谈文本进行归纳式主题分析。

请从以下访谈片段中提取编码。

要求：
- 识别文本中有意义的经历、行为、观点和规律
- 一个访谈片段可能包含多个意义单元，请分别识别
- 每条编码只对应一个表达完整的观点、经历、行为或理由，并用简短词组或短句表达
- 不要预设文本未说明的人群、场景或研究主题
- 编码内容使用与访谈文本相同的语言，不编造文本中没有的信息
- 返回包含 "codes" 数组的 JSON 对象，每条编码包含 "description" 字段

访谈片段：
{chunk_text}

只返回符合以下结构的 JSON 对象，不添加解释或 Markdown：
{{
  "codes": [
    {{"description": "第一条编码的简短描述"}},
    {{"description": "第二条编码的简短描述"}}
  ]
}}
"""


def build_theme_generation_prompt(code_descriptions: Iterable[str]) -> str:
    codes_text = "\n".join(f"- {description}" for description in code_descriptions)
    return f"""你是一名质性研究者，正在进行归纳式主题分析。

请将以下编码归纳为连贯的主题。

要求：
- 将相关编码归入更宽泛的主题
- 每个主题有清楚、具体的名称，描述用一句简短的话表达
- 主题应概括材料中的重要规律，并与其他主题有所区分
- 每个主题的 "codes" 字段列出其包含的原始编码描述，不改写这些描述
- 仅依据提供的编码归纳主题，不编造其中未出现的人群、场景或研究主题
- 主题名称和描述使用与编码相同的语言

编码：
{codes_text}

只返回符合以下结构的 JSON 对象，不添加解释或 Markdown：
{{
  "themes": [
    {{
      "name": "主题名称",
      "description": "用一句简短的话描述主题",
      "codes": ["属于该主题的原始编码描述"]
    }}
  ]
}}
"""


def build_evaluation_question_instructions(criteria: Any) -> Dict[str, str]:
    return {
        "coverage": (
            f"评估主题的覆盖度：{criteria.coverage}。只依据材料评分，不编造依据。"
        ),
        "actionability": (
            f"评估主题的概念清晰度：{criteria.actionability}。"
        ),
        "distinctiveness": (
            f"评估主题的区分度：{criteria.distinctiveness}。结合其他主题判断重叠程度。"
        ),
        "relevance": (
            f"评估主题的相关性：{criteria.relevance}。"
            "编码不足以支持判断时给出较低评分，并指出证据缺口。"
        ),
        "needs_refinement": (
            "判断该主题是否需要修订。请独立检查四项标准："
            f"覆盖度——{criteria.coverage}；"
            f"概念清晰度——{criteria.actionability}；"
            f"区分度——{criteria.distinctiveness}；"
            f"相关性——{criteria.relevance}。"
            "任一重要标准未达到良好水平时回答‘是’。"
        ),
    }


def build_evaluation_feedback_prompt(
    *,
    theme: Mapping[str, Any],
    other_themes: Sequence[Mapping[str, Any]],
    original_codes: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
) -> str:
    other_themes_text = "\n".join(
        f"- {item['name']}: {item['description']}" for item in other_themes
    )
    codes_text = "\n".join(f"- {code['description']}" for code in original_codes)
    theme_codes_text = "\n".join(f"- {code}" for code in theme.get("codes", []))
    return f"""你是一名质性研究者，正在为主题评估撰写具体反馈。
评分已由评估流程给出，直接采用，不重新评分。

主题「{theme['name']}」的评分结果：
- 覆盖度：{scores['coverage']:.2f}/5
- 概念清晰度：{scores['actionability']:.2f}/5
- 区分度：{scores['distinctiveness']:.2f}/5
- 相关性：{scores['relevance']:.2f}/5

待评估主题：
名称：{theme['name']}
描述：{theme['description']}
关联编码：
{theme_codes_text}

其他主题（用于比较区分度）：
{other_themes_text}

原始编码：
{codes_text}

要求：
- 每项标准的反馈需解释评分依据，低于 4 分的标准给出改进建议
- 如果编码不足以支持某项判断，指出证据缺口，不编造依据
- 重新判断该主题是否需要修订，并给出具体修订建议列表
- 反馈使用与主题和编码相同的语言

只返回符合以下结构的 JSON 对象，不添加解释或 Markdown；保留英文键名：
{{
  "coverage_feedback": "覆盖度的具体反馈",
  "actionability_feedback": "概念清晰度的具体反馈",
  "distinctiveness_feedback": "区分度的具体反馈",
  "relevance_feedback": "相关性的具体反馈",
  "needs_refinement": false,
  "refinement_suggestions": []
}}
"""


def build_full_evaluation_prompt(
    *,
    criteria: Any,
    theme: Mapping[str, Any],
    other_themes: Sequence[Mapping[str, Any]],
    original_codes: Sequence[Mapping[str, Any]],
) -> str:
    other_themes_text = "\n".join(
        f"- {item['name']}: {item['description']}" for item in other_themes
    )
    codes_text = "\n".join(f"- {code['description']}" for code in original_codes)
    theme_codes_text = "\n".join(f"- {code}" for code in theme.get("codes", []))
    return f"""你是一名质性研究者，正在评估访谈分析形成的主题。
仅使用提供的主题和编码，不预设材料未说明的人群或研究主题。

请依据以下四项标准评估主题：

1. 覆盖度：{criteria.coverage}
2. 概念清晰度：{criteria.actionability}
3. 区分度：{criteria.distinctiveness}
4. 相关性：{criteria.relevance}

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


def build_refinement_prompt(
    *,
    themes: Sequence[Mapping[str, Any]],
    theme_evaluations: Sequence[Mapping[str, Any]],
    global_feedback: str,
    codes: Sequence[Mapping[str, Any]],
) -> str:
    themes_text = json.dumps(themes, ensure_ascii=False, indent=2)
    evaluations_text = json.dumps(theme_evaluations, ensure_ascii=False, indent=2)
    codes_text = "\n".join(f"- {code['description']}" for code in codes)
    return f"""你是一名质性研究者，正在根据评估反馈修订主题。

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
{global_feedback}

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


def build_llm_decision_prompt(state: Any, questions: Mapping[str, Any]) -> str:
    lines = [
        "你是一名质性研究者。仅依据提供的材料回答问题，不引入材料之外的信息，也不编造依据。",
        "",
        "材料（JSON）：",
        json.dumps(state, ensure_ascii=False, indent=2),
        "",
        "请依次回答以下问题：",
    ]
    for question in questions.values():
        lines.append(
            f"- 「{question.key}」（{_decision_kind_label(question)}）：{question.instructions}"
        )
        if question.kind == "score" and question.scale:
            levels = "；".join(
                f"{index}. {level}" for index, level in enumerate(question.scale)
            )
            lines.append(f"  等级从低到高：{levels}")
        if question.kind == "choice" and question.options:
            options = "；".join(
                f"{name}——{description}" for name, description in question.options.items()
            )
            lines.append(f"  可选答案：{options}")
    lines.extend([
        "",
        "每项判断还需给出 confidence（0 到 1 的小数），表示你对该项判断的把握。",
        "",
        "只返回符合以下结构的 JSON 对象，不添加解释或 Markdown；保留英文键名：",
        "{",
        '  "answers": {',
    ])
    for question in questions.values():
        lines.append(f'    "{question.key}": {_decision_example_answer(question)},')
    lines.extend([
        "  }",
        "}",
    ])
    return "\n".join(lines)


def _decision_kind_label(question: Any) -> str:
    return {
        "noul": "是非判断（probability 为 0 到 1 的数，表示答案为「是」的可能性）",
        "choice": "选择（choice 从可选答案中选出一个）",
        "score": "评分（score 为等级序号，从 0 开始）",
    }[question.kind]


def _decision_example_answer(question: Any) -> str:
    if question.kind == "noul":
        return '{"probability": 0.8, "confidence": 0.7}'
    if question.kind == "choice":
        return '{"choice": "选项名称", "confidence": 0.85}'
    return '{"score": 3, "confidence": 0.9}'
