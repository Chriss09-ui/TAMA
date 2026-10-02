# Threadline Framework - Detailed Usage Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Usage](#basic-usage)
3. [Advanced Configuration](#advanced-configuration)
4. [Understanding the Workflow](#understanding-the-workflow)
5. [Interpreting Results](#interpreting-results)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Introduction

Threadline automates qualitative thematic analysis of already collected interview transcripts using three specialized AI agents:

- **Generation Agent**: Creates initial themes from transcript
- **Evaluation Agent**: Assesses theme quality using four criteria
- **Refinement Agent**: Improves themes based on feedback

Each submission is an independent analysis that produces one report. After initial coding, a single full-text review checks missed support, counterexamples and context limits before the existing evaluation cycle. The web interface does not provide cross-run continuation or sampling tasks.

## Basic Usage

### Step 1: Prepare Your Transcript

Your transcript should be a plain text file containing interview content. Example format:

```
Interview with Participant A:

Interviewer: Can you tell me about...

Participant A: [Response]

Interview with Participant B:

...
```

Place your transcript in the `data/` directory.

### Step 2: Run Analysis

```python
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from tama import ThreadlineFramework, load_transcript

# Initialize
framework = ThreadlineFramework(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o"
)

# Load transcript
transcript = load_transcript("data/your_transcript.txt")

# Run analysis
result = framework.run_analysis(
    transcript=transcript,
    session_name="my_analysis"
)
```

### Step 3: Review Results

Check `outputs/my_analysis/` for:
- `00_summary.txt`: Human-readable summary
- `00_final_results.json`: Complete data

## Advanced Configuration

### Configuration Parameters

```python
framework = ThreadlineFramework(
    api_key="your-api-key",

    # Model selection
    model="gpt-4o",  # Options: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"

    # Quality control
    acceptance_threshold=4.0,  # Min score (1.0-5.0) to accept themes
    max_iterations=5,          # Max refinement cycles
    corpus_review=True,        # One full-text review; enabled by default

    # Output settings
    output_dir="outputs",      # Where to save results

    # Custom evaluation criteria (optional)
    expert_criteria={
        "coverage": "Custom coverage criterion...",
        "actionability": "Custom actionability criterion...",
        "distinctiveness": "Custom distinctiveness criterion...",
        "relevance": "Custom relevance criterion..."
    }
)
```

### Model Selection Guidelines

| Model | Speed | Cost | Quality | Best For |
|-------|-------|------|---------|----------|
| gpt-4o | Fast | Medium | High | Production use (recommended) |
| gpt-4o-mini | Very Fast | Low | Good | Testing, quick analysis |
| gpt-4-turbo | Medium | High | Highest | Maximum quality needed |

### Acceptance Threshold

The acceptance threshold determines when themes are good enough:

- **3.0-3.5**: Lenient - faster completion, lower quality
- **4.0**: Balanced - recommended for most use cases
- **4.5-5.0**: Strict - may require many iterations

### Custom Evaluation Criteria

Tailor evaluation to your domain:

```python
# For mental health interviews
mental_health_criteria = {
    "coverage": "should capture all reported emotional states and coping strategies",
    "actionability": "should focus on a single psychological concept or experience",
    "distinctiveness": "should not overlap with other psychological themes",
    "relevance": "must reflect the patient's subjective experience accurately"
}

# For educational research
education_criteria = {
    "coverage": "should encompass all learning challenges and successes mentioned",
    "actionability": "should represent a clear educational concept or practice",
    "distinctiveness": "should be pedagogically distinct from other themes",
    "relevance": "must align with teachers' and students' reported experiences"
}

framework = ThreadlineFramework(
    api_key=api_key,
    expert_criteria=mental_health_criteria  # or education_criteria
)
```

## Understanding the Workflow

### Phase 1: Generation

**Input**: Raw transcript (any length)

**Steps**:
1. **Chunking**: Selects a dynamic target from total text length and cuts at natural transcript boundaries
2. **Coding**: Extracts codes (<25 words) from each chunk
3. **Theme Generation**: Synthesizes codes into themes (~25 words)

**Output**: Initial themes with associated codes

**Duration**: ~2-5 minutes for typical interviews

### Phase 2: Evaluation

**Input**: Generated themes + original codes

**Process**: Evaluates each theme on 4 criteria (1-5 scale):
- **Coverage**: Does it capture important patterns comprehensively?
- **Actionability**: Is it a single, clear concept?
- **Distinctiveness**: Is it clearly distinct from other themes?
- **Relevance**: Does it accurately reflect the data?

**Output**:
- Individual theme scores
- Average score across all themes
- Accept/reject decision
- Specific improvement suggestions

**Duration**: ~1-2 minutes

### Phase 3: Refinement

**Input**: Themes + evaluation feedback

**Operations**:
- **DELETE**: Remove irrelevant themes
- **COMBINE**: Merge overlapping themes
- **SPLIT**: Separate multi-concept themes
- **ADD**: Include missing important themes

**Output**: Refined themes

**Duration**: ~1-2 minutes

### Iteration

The framework repeats Evaluation → Refinement until:
- Themes achieve acceptance threshold, OR
- Maximum iterations reached

**Typical iterations**: 2-4 cycles

## Interpreting Results

### Final Results JSON Structure

```json
{
  "session_name": "my_analysis",
  "timestamp": "2025-01-16T10:30:00",
  "configuration": {...},
  "generation": {
    "num_chunks": 5,
    "num_codes": 47,
    "initial_num_themes": 8
  },
  "refinement_iterations": 3,
  "final_themes": [
    {
      "name": "Theme name",
      "description": "Theme description (~25 words)",
      "codes": ["Associated code descriptions..."]
    },
    ...
  ],
  "metadata": {
    "total_themes": 7,
    "final_average_score": 4.2
  },
  "accepted": true
}
```

### Theme Quality Indicators

**Good themes have**:
- Clear, descriptive names
- Concise descriptions (~25 words)
- 3+ associated codes
- Scores ≥4.0 on all criteria

**Warning signs**:
- Very long descriptions (>40 words) → may contain multiple concepts
- Only 1-2 codes → may be too specific or not well-supported
- Low distinctiveness score → may overlap with other themes

### Summary File Format

```
================================================================================
Threadline THEMATIC ANALYSIS - SUMMARY
================================================================================

Session: my_analysis
Timestamp: 2025-01-16T10:30:00
Model: gpt-4o
Status: ACCEPTED
Final Score: 4.2/5.0
Refinement Iterations: 3

================================================================================
FINAL THEMES
================================================================================

1. [Theme Name]
   [Theme description]
   Associated codes: 5

2. [Theme Name]
   [Theme description]
   Associated codes: 8

...

================================================================================
REFINEMENT HISTORY
================================================================================

Iteration 1:
  Evaluation Score: 3.5/5.0
  Refinement: Combined 2 overlapping themes, deleted 1 irrelevant theme
  Theme Count: 8 -> 6

Iteration 2:
  Evaluation Score: 3.9/5.0
  Refinement: Split 1 multi-concept theme, added 1 missing theme
  Theme Count: 6 -> 7

Iteration 3:
  Evaluation Score: 4.2/5.0
  Refinement: Minor adjustments to theme descriptions
  Theme Count: 7 -> 7
```

## Best Practices

### 1. Transcript Preparation

**Do**:
- Include speaker labels (Interviewer, Participant A, etc.)
- Maintain conversation flow
- Include relevant context

**Don't**:
- Include excessive metadata or timestamps
- Use special characters heavily
- Mix multiple languages extensively

### 2. Configuration Selection

**For exploratory analysis**:
```python
framework = ThreadlineFramework(
    model="gpt-4o-mini",
    acceptance_threshold=3.5,
    max_iterations=3
)
```

**For publication-quality analysis**:
```python
framework = ThreadlineFramework(
    model="gpt-4o",
    acceptance_threshold=4.5,
    max_iterations=7,
    expert_criteria=custom_criteria
)
```

### 3. Iterative Refinement

- Start with default settings
- Review first-pass themes
- Adjust criteria based on domain needs
- Re-run with custom criteria

### 4. Quality Validation

Always:
1. Review final themes manually
2. Check if themes make sense for your domain
3. Verify themes are supported by actual quotes
4. Compare with any existing manual analysis

### 5. Documentation

Save your configuration for reproducibility:

```python
config = {
    "date": "2025-01-16",
    "transcript": "parent_interviews_aaoca.txt",
    "model": "gpt-4o",
    "threshold": 4.0,
    "iterations": 5,
    "custom_criteria": expert_criteria
}

# Save with results
import json
with open("outputs/my_analysis/config.json", 'w') as f:
    json.dump(config, f, indent=2)
```

## Troubleshooting

### Issue: Low Quality Themes

**Symptoms**: Vague themes, low scores, doesn't capture data well

**Solutions**:
- Increase `max_iterations` to allow more refinement
- Provide custom `expert_criteria` specific to your domain
- Check if transcript is clear and well-formatted
- Try using `gpt-4o` instead of `gpt-4o-mini`

### Issue: Too Many Iterations

**Symptoms**: Reaches max iterations without acceptance

**Solutions**:
- Lower `acceptance_threshold` (e.g., from 4.0 to 3.5)
- Review custom criteria - may be too strict
- Check if transcript quality is sufficient
- Increase `max_iterations` if themes are improving each cycle

### Issue: Too Few Themes

**Symptoms**: Only 2-3 themes generated, missing important patterns

**Solutions**:
- Check transcript length - may be too short
- Review Generation Agent's codes - are patterns being captured?
- Explicitly add coverage criterion emphasizing comprehensiveness
- Manually review and provide feedback for additional iteration

### Issue: Too Many Themes

**Symptoms**: 15+ themes, many seem overlapping

**Solutions**:
- Increase `acceptance_threshold` to force more combining
- Emphasize distinctiveness in custom criteria
- Review Refinement Agent's COMBINE operations
- Manually consolidate similar themes in final review

### Issue: API Errors

**Symptoms**: Rate limit errors, timeouts

**Solutions**:
- Use `gpt-4o-mini` to reduce rate limit pressure
- Add delays between iterations (modify source code)
- Select **自动 · 精细** or lower the manual chunk limit for smaller API calls
- Check OpenAI account tier and limits

### Issue: Inconsistent Results

**Symptoms**: Different themes on repeated runs

**Solutions**:
- This is expected with LLMs - some variation is normal
- For more consistency, lower temperature (requires code modification)
- Run multiple times and compare results
- Use custom criteria to guide consistency

## Example Use Cases

### Clinical Interviews (AAOCA Parents)

```python
clinical_criteria = {
    "coverage": "should comprehensively capture parent experiences, concerns, and medical journey",
    "actionability": "should focus on a single aspect of the AAOCA experience",
    "distinctiveness": "should represent a unique dimension of parent perspectives",
    "relevance": "must accurately reflect parents' reported experiences and emotions"
}

framework = ThreadlineFramework(
    api_key=api_key,
    model="gpt-4o",
    acceptance_threshold=4.0,
    expert_criteria=clinical_criteria
)
```

### User Research Interviews

```python
user_research_criteria = {
    "coverage": "should capture all user pain points, needs, and behaviors",
    "actionability": "should represent a distinct user need or behavior pattern",
    "distinctiveness": "should be clearly different from other user themes",
    "relevance": "must reflect actual user statements and experiences"
}

framework = ThreadlineFramework(
    api_key=api_key,
    model="gpt-4o-mini",  # Faster for iterative research
    acceptance_threshold=3.5,
    expert_criteria=user_research_criteria
)
```

### Educational Research

```python
education_criteria = {
    "coverage": "should encompass all teaching challenges and successes",
    "actionability": "should represent a clear pedagogical concept",
    "distinctiveness": "should be educationally distinct from other themes",
    "relevance": "must align with educators' reported experiences"
}

framework = ThreadlineFramework(
    api_key=api_key,
    model="gpt-4o",
    acceptance_threshold=4.5,  # Higher standard for publication
    expert_criteria=education_criteria
)
```

## Additional Resources

- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md)
- **Example Script**: [example_usage.py](example_usage.py)

## Evidence and full-text review

Only codes whose excerpts and character offsets match the submitted original are eligible for generation and scoring. Unmatched or missing excerpts remain in the result as pending records. A genuine participant statement remains eligible even when it has not been externally verified.

Full-text review runs once per submission and can be disabled with corpus_review=False. Verified new findings are added and candidate themes updated automatically. Review failures are explicit; a partial or failed enabled review cannot pass the overall analysis. An analysis with no valid evidence returns a savable result with final_average_score set to null, displayed as unscored.

The complete JSON includes source_documents (normalized text, source ID and SHA-256), corpus_review (status, chunk counts, added IDs and findings), and code evidence_status. Word does not include the entire original. Set both save_final=False and save_intermediate=False to avoid creating an output directory.

## Comparison and researcher review

The result page offers a group-by-theme matrix using supplied participant, case, speaker, event, time or source information. Support, opposition, mixed evidence and no linked evidence remain distinct. Speaker labels do not establish participant identity, and missing identifiers remain unknown. Counts distinguish codes, original-text positions, sources, participants and cases.

Save each theme interpretation explicitly. Stable theme_id values keep saved notes attached when themes are reordered. Saved notes enter JSON, text and Word reports. Thematic mode also offers the overall argument editor. Two alternative explanations can separately reference verified supporting and contradicting codes, boundaries and unresolved questions. These edits do not call a model.

Researcher review is separate from automated acceptance. Editing reviewed interpretations, arguments or evidence groups resets review to pending. Counterexamples, information breakpoints and evidence gaps are retained for review rather than scored or refined as shared patterns. If no scorable pattern remains, the final score is null; earlier scores remain in score_history.

Expand a quote's original question/answer unit or paragraph using the saved original and offsets. Optional report context contains selected quote contexts; long contexts are shortened with an explicit notice. Missing originals produce a labelled fallback, not reconstructed text.

Restore complete JSON through “查看已保存报告（JSON）” for local report viewing and editing. This rechecks originals and evidence, restores notes, and disables disk paths from the imported file. It never starts or continues analysis. Download again to preserve restored edits. New submissions remain independent.

Additional result fields: analysis_id, final_themes[].theme_id, researcher_annotations (theme_notes, explanations, review, source_versions), report_options and audit_trail. Audit events record analysis milestones and human changes; missing legacy history is not invented. Saved reports and their audit trails contain researcher interpretations, so treat them as research data. Private memos and reflexivity remain isolated from model prompts.
