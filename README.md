# TAMA: Human–AI Collaborative Thematic Analysis using Multi-Agent LLMs

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![LLM](https://img.shields.io/badge/LLM-GPT--4o-black.svg)
![Status](https://img.shields.io/badge/ACM%20Computing%20for%20Healthcare-2025-blue.svg)

## Overview

This repository implements a human–AI workflow for qualitative interview analysis. Generation, Evaluation, and Refinement agents extract grounded codes, propose themes, assess them, and revise them. The default profile is topic-neutral. An optional enterprise-evidence profile supplies a specific research question and focus areas. Model outputs and scores require researcher review. Quantitative metrics reported in the TAMA paper are not implemented in this repository.

## Quick Start

### Continuing a study

The input section's **来源与持续研究** panel records source context and can
continue the current result or import an earlier complete JSON result. New
codes keep distinct source identities and receive IDs after the previous
round's IDs. Identical labels no longer cause automatic merging of distinct
evidence; merge prompts compare the original excerpts and context.

Select **建构扎根理论支持** to work with researcher-selected focused codes,
category properties and boundaries, theoretical sampling tasks, category
sufficiency judgements, relationships, arguments, and the four study review
criteria. Comparisons and analytic decisions are also available in the default
thematic workflow. Saving records does not call the model. Only decisions
explicitly selected and confirmed for a subsequent round enter its prompts;
private memos and reflexivity remain separate.

Complete JSON and Word downloads include the research records. With local
saving enabled, `08_research_records.md` accompanies the result, and editor
saves update the local JSON, codebook, next-data plan, and summary. Theme
scores and processing completion do not determine theoretical saturation.

The methodological mapping and the archived assessment are in
[建构扎根理论对照](建构扎根理论对照/README.md), alongside the existing
[编码手册对照](编码手册对照/项目如何符合编码手册.md).

**Get started in 5 minutes!** See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY='your-api-key'
python example_usage.py
```

### Xiaomi MiMo API

The example also supports Xiaomi MiMo through its OpenAI-compatible API. Set
`MIMO_API_KEY` to use MiMo instead of OpenAI. The default MiMo model is
`mimo-v2.5-pro`, which supports the JSON output used by all three agents.

```bash
export MIMO_API_KEY='your-mimo-api-key'
python example_usage.py
```

For a MiMo Token Plan, set `MIMO_BASE_URL` to the OpenAI-compatible Base URL
provided for your plan. You can also set `MIMO_MODEL` to another model that
supports JSON output, such as `mimo-v2.5`. When multiple provider keys are set,
the example uses MiMo first. For direct `TAMAFramework` use, pass the MiMo API key,
`model="mimo-v2.5-pro"`, and
`base_url="https://api.xiaomimimo.com/v1"` (or your Token Plan Base URL).

See the [MiMo first API call](https://mimo.mi.com/docs/zh-CN/quick-start/summary/first-api-call)
and [structured output](https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/text-generation/structured-output)
documentation for current endpoints and supported models.

### DeepSeek API

DeepSeek offers an OpenAI-compatible API and JSON output for the analysis
agents. Set `DEEPSEEK_API_KEY` to use the DeepSeek-V4.1-Flash model through
its API model ID `deepseek-flash` and official OpenAI-compatible endpoint
`https://api.deepseek.com`:

```bash
export DEEPSEEK_API_KEY='your-deepseek-api-key'
python example_usage.py
```

You can override the defaults with `DEEPSEEK_MODEL` or `DEEPSEEK_BASE_URL`.
When multiple provider keys are set, `example_usage.py` selects MiMo, then
DeepSeek, then OpenAI. The web interface starts with DeepSeek selected and
shows `DeepSeek-V4.1-Flash` as its model name; requests use `deepseek-flash`.
Enter the API key in the sidebar or set `DEEPSEEK_API_KEY`. See the [DeepSeek first API call](https://api-docs.deepseek.com/zh-cn/)
and [JSON output](https://api-docs.deepseek.com/zh-cn/guides/json_mode/)
documentation for the current model names and endpoint.

### Local web interface

Install the dependencies and launch the local interface:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The interface accepts pasted text, a UTF-8 `.txt` file, or a `.docx` file.
DOCX body paragraphs and table rows are read in document order; scanned pages
or images need OCR before analysis. Select MiMo,
DeepSeek, or OpenAI in the sidebar, then enter the API key or set the selected
provider's `MIMO_API_KEY`, `DEEPSEEK_API_KEY`, or `OPENAI_API_KEY` environment
variable. Results appear on the page. Local output is optional and off by
default in the web interface. The app listens on `127.0.0.1` for local use.

After analysis, use **保存 Word 报告（DOCX）** to download a readable report with
the final themes, descriptions, and associated codes. Use **保存完整数据（JSON）**
to download the complete result for later processing. These downloads contain
verbatim excerpts when present. Enable **保存本地结果** to write
`00_final_results.json` and `00_summary.txt` under `outputs/`.
For no local transcript-derived files, leave both local saving and stage files
off. The transcript still goes to the selected model service and the result
remains in the local service process until it restarts or a new run replaces it.
Use **清除服务内结果** after downloading to discard the latest in-memory
result without restarting the service.

To keep an API key across browser and app restarts, enter it in the sidebar and
click **保存 API Key**. The key is stored in the operating system credential store
(macOS Keychain on macOS), separately for each provider, rather than in the
repository or `outputs/`. Leave the field empty to use the saved key; click
**删除已保存 Key** to remove it. A newly typed key takes precedence over a saved
key, and a saved key takes precedence over the provider's environment variable.

Use **测试 API 连接** in the sidebar to send one short request with the current
key, model, and endpoint; the provider may charge for it. During analysis,
**暂停分析** waits for any request already in progress to finish, then stops
before the next model request. **继续分析** resumes the same run. **取消分析**
stops at the next request boundary. Refreshing the browser reconnects to the
running job and its result; keep the local Streamlit service running. An
already-sent request may finish and be charged before pause or cancellation.

Choose **切块策略** in the sidebar before starting analysis. The default
**自动 · 均衡** mode calculates a document-specific target from the complete
transcript length, then places boundaries at complete question-answer groups,
paragraphs, speaker turns, or sentence endings. **自动 · 精细** uses smaller
chunks, while **自动 · 少调用** uses larger chunks. **手动设置** preserves the
previous adjustable limit, counting Chinese characters individually and other
text by whitespace-separated words. The page previews the expected chunk count
before analysis. Final JSON records the selected strategy, resolved target,
hard limit, source length, and actual chunk count.

Independent code extraction requests and theme evaluations run concurrently.
The sidebar's **并发请求数** defaults to 4 and can be set from 1 to 8; lower it
if the model provider rate-limits requests. Results keep the original chunk
and theme order. Theme generation still waits for all codes, and refinement
waits for all evaluations. Pausing blocks requests that have not started yet;
requests already sent to the provider may finish first. The selected limit is
recorded as `configuration.max_workers` in the final JSON result.

### Decision model integration (System One / Jev)

Theme evaluation uses a hybrid two-stage flow. A decision provider first
answers typed questions for each theme (four 1–5 criterion scores plus a
yes/no refinement decision). Score answers include confidence; Jev's Noul
answer supplies only the probability of "yes", so the app derives a separate
certainty proxy from its distance to 0.5. Themes that pass
every criterion with confident decisions skip the expensive feedback call;
themes with any score below 4, a refinement decision, or a confidence below
the threshold get a second call that writes feedback text and refinement
suggestions. Decisions with low confidence are marked `建议人工复核` in the
results.

Two decision providers are available:

- **跟随主模型 (default)**: the selected analysis model answers the typed
  questions in JSON mode. Confidence values are self-reported by the model —
  a weak signal kept for the audit trail, not a calibrated probability.
- **Jev（实验性）**: the TypeSafe Jev System One decision API
  ([docs](https://docs.typesafe.ai/api)) returns probability-based scores and
  confidence in a single request per theme. Select this mode and enter the
  Jev API key in the sidebar; it can be stored in the system credential vault.
  `JEV_API_KEY` remains available as a fallback, with optional `JEV_BASE_URL`
  and `JEV_MODEL`. The sidebar's **测试 API 连接** verifies the decision
  endpoint separately. Jev evaluates themes; the main text model still
  extracts codes, generates themes, and writes detailed feedback. Jev does
  not generate the qualitative analysis text. TypeSafe notes that accuracy
  on Chinese text needs validation with the user's own examples.

The sidebar's **决策模式** selects the provider. The confidence threshold
lives under **高级设置** (default 0.7). Saved results record the provider and
threshold under `configuration.decision_provider` /
`configuration.confidence_threshold`, and each theme evaluation carries
`score_confidences`, zero-based `raw_scores`, one-based probability-weighted
`weighted_scores`, `flagged_for_review`, and
`feedback_source` (`llm`, `placeholder`, or `fallback`) under
`final_evaluation.theme_evaluations` and, when stage files are enabled, in
`02_evaluation_iter*.json`. The run is accepted only when the average and
every criterion for every theme meet the threshold and no theme needs
refinement. A positive typed refinement decision cannot be cancelled by the
main model's prose feedback. If a decision request fails with a recoverable
error, the theme falls back to the legacy single-call evaluation and the
result is marked `fallback`. Jev configuration errors (for example an
invalid key) stop the run instead of being silently ignored.

## Key Features

- Multi-agent LLM architecture with coordinated Generation, Evaluation, and Refinement agents.
- Researcher-configured study focus and evaluation criteria; final interpretation remains with the researcher.
- Decision-model integration: theme scores carry per-criterion confidence; low-confidence themes are flagged for human review.
- Centralized prompts: all model instructions, evaluation rubrics, and connection-test questions live in `src/prompts.py`.
- Duplicate-code compaction, batched theme synthesis, and score-history early stopping.

## Architecture

The TAMA framework uses a multi-agent workflow consisting of Generation, Evaluation, and Refinement agents, with iterative oversight from a clinical expert. The expert provides initial goals and evaluation criteria and makes the final decision to accept or revise the generated themes.

### Workflow Diagram
![TAMA Architecture](figure/TAMA-Workflow.png)

## Installation

### Prerequisites
- Python 3.10 or higher
- An API key for MiMo, DeepSeek, or OpenAI

### Quick Setup

```bash
# Run from the project directory
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY='your-api-key-here'

# Run example
python example_usage.py
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

## Usage

### Basic Example

```python
import os
from src.tama import TAMAFramework, load_transcript

# Initialize framework
tama = TAMAFramework(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    max_iterations=5,
    acceptance_threshold=4.0
)

# Load transcript and run analysis
transcript = load_transcript("data/your_transcript.txt")
result = tama.run_analysis(
    transcript=transcript,
    session_name="my_analysis",
    save_intermediate=True
)

# Access results
print(f"Generated {len(result['final_themes'])} themes")
print(f"Final score: {result['metadata']['final_average_score']:.2f}/5.0")
```

### Custom Evaluation Criteria

The runtime instructions are written in Chinese. Generated codes, themes, and
feedback follow the language of the supplied interview text. The default
`generic` profile does not assume a topic. Select `enterprise_evidence` to
study specific capabilities, internal evidence, holders, public information,
and information gaps. Both profiles accept `research_question`, `focus_areas`,
and `expert_criteria`; focus areas are guidance, not a requirement to find
them in every interview. The bundled sample transcript is a clinical example.

```python
# Define study-specific criteria from researchers
expert_criteria = {
    "coverage": "应覆盖材料中的重要规律",
    "actionability": "应表达一个清楚、便于理解的概念",
    "distinctiveness": "应与其他主题明确区分",
    "relevance": "应准确反映受访者的表达，并有材料支持"
}

tama = TAMAFramework(
    api_key=api_key,
    profile="generic",  # or "enterprise_evidence"
    research_question="受访者如何理解这段经历？",
    focus_areas=["决策过程", "不确定性"],
    expert_criteria=expert_criteria
)
```

### Output Structure

`TAMAFramework.run_analysis` saves final files by default for API callers,
while stage-file saving defaults to off. Pass `save_final=False` for an
in-memory run. Saved files keep verbatim excerpts. When enabled, files under
`outputs/[session_name]/` are:

- `00_final_results.json` and `00_summary.txt`: final result
- `04_memos.md` and `04_memos.json`: why each theme holds those codes. Text between `[human]` markers is for the researcher and is not sent to the model
- `04_code_memos.md`: code-level memos written before themes. Researcher notes in the same markers are not sent to the model
- `04_memos_iter*.json`: the same theme memo at each generation or refinement pass
- `05_codebook.md`: method, definition, inclusion, exclusion, related codes, notes, and examples for each canonical code
- `06_code_map.md`: four archived steps from the full code list to categories and concepts
- `07_code_landscape.md`: outline of canonical codes with excerpt counts. Counts are not importance
- `next_data_plan.md`: open questions grouped for the next round of data collection
- `01_generation.json`: chunks and codes, only with `save_intermediate=True`
- `02_evaluation_iter*.json`, `03_refinement_iter*.json`: iterative audit files, only with `save_intermediate=True`

Names generated automatically include a random suffix. A supplied name that
already exists receives a suffix instead of overwriting an earlier run.

## Framework Components

### 1. Generation Agent
- **Chunking**: Dynamically sizes model windows and preserves natural transcript boundaries
- **Coding**: Extracts codes (<25 words) from each chunk
- **Theme Generation**: Collapses identical labels, writes definitions at that merge, maps codes into categories, synthesizes at most 40 distinct codes per batch, then consolidates candidate themes and records a draft analytic storyline

### 2. Evaluation Agent
Evaluates themes using four criteria:
- **Coverage**: Comprehensively captures important patterns
- **Pattern** (stored as `actionability`): a shared meaning across the material, rather than a topic, background section, or interview-guide heading. A background section scores 1 or 2; a pattern with supporting codes scores 4.
- **Distinctiveness**: Clearly distinct from other themes
- **Relevance**: Accurately reflects the data

### 3. Refinement Agent
Refines themes using four operations:
- **Add**: Add missing important themes
- **Split**: Split themes containing multiple concepts
- **Combine**: Combine overlapping themes
- **Delete**: Delete irrelevant themes

### Iterative Process
The framework iterates through evaluation and refinement until:
- Themes achieve acceptance threshold (default: 4.0/5.0), OR
- Maximum iterations reached (default: 5)
- Average score fails to improve by more than 0.05 for two consecutive evaluations (configurable with `early_stop_patience`)

Each result records `score_history` and `stop_reason`. Scores are model
judgments, not calibrated measures of qualitative validity.

### Run tests

From this checkout, use `.venv/bin/python -m unittest discover -s tests`.
The existing `.venv` was created without `pip`; if dependencies need to be
installed into it, use `uv pip install --python .venv/bin/python -r requirements.txt`.
