# TAMA Framework - Implementation Summary

## Overview

This document provides a technical summary of the TAMA (Thematic Analysis using Multi-Agent LLMs) framework implementation.

**Implementation Date**: January 16, 2025
**Version**: 1.0.0
**Language**: Python 3.10+
**Primary LLM**: OpenAI GPT-4o
**Framework Type**: Multi-Agent System

---

## Implementation Statistics

| Component | Lines of Code | File Size |
|-----------|--------------|-----------|
| Generation Agent | ~270 | 7.8 KB |
| Evaluation Agent | ~350 | 11 KB |
| Refinement Agent | ~320 | 10 KB |
| Main Orchestrator | ~240 | ~8 KB |
| **Total Core Code** | **~1,180** | **~37 KB** |

### Documentation

| Document | Purpose | Size |
|----------|---------|------|
| README.md | Main documentation | 3.0 KB |
| QUICKSTART.md | 5-minute guide | 5.8 KB |
| INSTALLATION.md | Setup instructions | 6.2 KB |
| USAGE_GUIDE.md | Comprehensive guide | 14.5 KB |
| PROJECT_STRUCTURE.md | Project organization | 6.8 KB |
| **Total Documentation** | | **~36 KB** |

---

## Architecture

### Multi-Agent System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     TAMA Framework                          │
│                    (tama.py - Orchestrator)                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌───────────────┐
│  Generation   │    │   Evaluation   │    │  Refinement   │
│     Agent     │───▶│     Agent      │───▶│     Agent     │
│               │    │                │    │               │
│ • Chunking    │    │ • Coverage     │    │ • Add         │
│ • Coding      │    │ • Actionable   │    │ • Split       │
│ • Themes      │    │ • Distinctive  │    │ • Combine     │
│               │    │ • Relevance    │    │ • Delete      │
└───────────────┘    └────────────────┘    └───────────────┘
                              │
                              └──────────┐
                              Iteration  │
                              Loop       │
                              ▲          │
                              └──────────┘
```

---

## Component Implementation Details

### 1. Generation Agent (`generation_agent.py`)

**Purpose**: Transform raw transcripts into initial themes

**Key Classes**:
- `Chunk`: Represents dynamically sized transcript windows with source offsets
- `Code`: Extracted patterns (<25 words)
- `Theme`: Synthesized themes (~25 words)
- `GenerationAgent`: Main agent class

**Methods**:
```python
chunk_transcript(transcript: str) -> List[Chunk]
generate_codes_from_chunk(chunk: Chunk) -> List[Code]
generate_codes(chunks: List[Chunk]) -> List[Code]
generate_themes(codes: List[Code]) -> List[Theme]
run(transcript: str) -> Dict[str, Any]
```

**LLM Calls**:
- N chunks + 1 (for coding + theme generation)
- Temperature: 0.3
- Format: JSON structured output

**Output**: Chunks, codes, initial themes

---

### 2. Evaluation Agent (`evaluation_agent.py`)

**Purpose**: Assess theme quality using four criteria

**Key Classes**:
- `EvaluationCriteria`: Defines four evaluation criteria
- `EvaluationResult`: Per-theme evaluation scores
- `OverallEvaluation`: Aggregate evaluation
- `EvaluationAgent`: Main agent class

**Evaluation Criteria** (1-5 scale):
1. **Coverage**: Comprehensiveness of pattern capture
2. **Actionability**: Single, clear concept
3. **Distinctiveness**: Uniqueness among themes
4. **Relevance**: Accuracy to source data

**Methods**:
```python
evaluate_theme(theme, all_themes, codes) -> EvaluationResult
evaluate_all_themes(themes, codes) -> OverallEvaluation
run(themes, codes) -> Dict[str, Any]
```

**LLM Calls**:
- 1 per theme (evaluates each independently)
- Temperature: 0.2 (more deterministic)
- Format: JSON structured output

**Decision Logic**:
- Accept if: average_score >= acceptance_threshold (default: 4.0)
- Otherwise: proceed to refinement

---

### 3. Refinement Agent (`refinement_agent.py`)

**Purpose**: Improve themes based on evaluation feedback

**Key Classes**:
- `RefinementOperation`: Single refinement action
- `RefinementPlan`: Collection of operations
- `RefinementAgent`: Main agent class

**Refinement Operations**:
1. **DELETE**: Remove irrelevant themes (low relevance)
2. **COMBINE**: Merge overlapping themes (low distinctiveness)
3. **SPLIT**: Separate multi-concept themes (low actionability)
4. **ADD**: Include missing themes (low coverage)

**Methods**:
```python
create_refinement_plan(themes, eval_results, codes) -> RefinementPlan
apply_refinement_plan(themes, plan) -> List[Theme]
_apply_delete/combine/split/add(themes, operation) -> List[Theme]
run(themes, eval_results, codes) -> Dict[str, Any]
```

**LLM Calls**:
- 1 per refinement cycle (creates comprehensive plan)
- Temperature: 0.3
- Format: JSON structured output

**Operation Priority**: DELETE → COMBINE → SPLIT → ADD

---

### 4. TAMA Orchestrator (`tama.py`)

**Purpose**: Coordinate agents and manage workflow

**Key Class**: `TAMAFramework`

**Workflow**:
```python
1. Initialize agents (Generation, Evaluation, Refinement)
2. Run Generation Agent → initial themes
3. Iteration loop (max 5 by default):
   a. Run Evaluation Agent → scores & feedback
   b. Check if acceptable (avg_score >= threshold)
   c. If acceptable → STOP
   d. If not → Run Refinement Agent → improved themes
   e. Repeat from 3a
4. Save final results + metadata
```

**Configuration Parameters**:
- `model`: LLM model ("gpt-4o", "gpt-4o-mini", etc.)
- `max_iterations`: Maximum refinement cycles (default: 5)
- `acceptance_threshold`: Minimum score (default: 4.0/5.0)
- `output_dir`: Results directory (default: "outputs")
- `expert_criteria`: Optional custom evaluation criteria

**Methods**:
```python
__init__(api_key, model, max_iterations, ...)
run_analysis(transcript, session_name) -> Dict[str, Any]
_save_readable_summary(session_dir, result)
```

**Output Files**:
- `00_final_results.json`: Complete analysis data
- `00_summary.txt`: Human-readable summary
- `01_generation.json`: Generation output
- `02_evaluation_iterN.json`: Evaluation per iteration
- `03_refinement_iterN.json`: Refinement per iteration

---

## Data Flow

```
Input Transcript (text)
        │
        ▼
┌──────────────────┐
│ Chunking         │  → Dynamic chunks at natural transcript boundaries
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ Coding           │  → Codes (<25 words each)
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ Theme Generation │  → Initial Themes (~25 words)
└──────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Iteration Loop                       │
│                                      │
│  ┌──────────────┐                   │
│  │ Evaluation   │ → Scores + Feedback│
│  └──────────────┘                   │
│         │                            │
│         ▼                            │
│  ┌──────────────┐                   │
│  │ Check Score  │─ Acceptable? ──┐  │
│  └──────────────┘                 │  │
│         │ Not Acceptable          │  │
│         ▼                          │  │
│  ┌──────────────┐                 │  │
│  │ Refinement   │ → Refined Themes│  │
│  └──────────────┘                 │  │
│         │                          │  │
│         └──────────────────────────┘  │
│                 │ Acceptable           │
└─────────────────┼──────────────────────┘
                  ▼
         Final Themes (output)
```

---

## Technical Specifications

### Dependencies

**Core**:
- `openai>=1.12.0`: OpenAI API client
- `pydantic>=2.0.0`: Data validation and parsing

**Optional**:
- `langchain`: Advanced text processing
- `nltk`: Natural language toolkit
- `pandas`: Data manipulation

### API Usage

**Estimated API Costs** (per analysis):
- Typical interview: 5,000-20,000 words
- Chunks: determined from transcript length and the selected strategy
- LLM calls: ~10-20 total
- Cost with GPT-4o: $0.10-0.50
- Cost with GPT-4o-mini: $0.01-0.05

**Rate Limits**:
- Sequential API calls (not parallel)
- Respects OpenAI rate limits
- Typical runtime: 5-10 minutes

### Performance

**Runtime** (typical interview):
- Generation: 2-5 minutes
- Evaluation: 1-2 minutes per iteration
- Refinement: 1-2 minutes per iteration
- **Total**: 5-10 minutes (3-4 iterations)

**Comparison to Manual TA**:
- Manual coding: ~30 hours
- TAMA: ~10 minutes
- **Time reduction**: 99%+

---

## Code Quality Features

### Type Safety
- Pydantic models for all data structures
- Type hints throughout codebase
- Runtime validation

### Error Handling
- API error handling
- JSON parsing validation
- File I/O error management

### Modularity
- Separate agent files
- Clear interfaces between components
- Easy to extend/modify

### Logging
- Print statements for user feedback
- Progress indicators
- Clear status messages

### Documentation
- Docstrings for all classes/methods
- Type annotations
- Inline comments where needed

---

## Extensibility

### Adding New Agents

1. Create new file in `src/agents/`
2. Define agent class with `run()` method
3. Import in `src/agents/__init__.py`
4. Integrate into orchestrator workflow

Example:
```python
# src/agents/validation_agent.py
class ValidationAgent:
    def __init__(self, api_key, model):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def run(self, themes):
        # Validation logic
        return validation_results
```

### Adding New Evaluation Criteria

Modify `EvaluationCriteria` in `evaluation_agent.py`:
```python
class EvaluationCriteria(BaseModel):
    coverage: str = "..."
    actionability: str = "..."
    distinctiveness: str = "..."
    relevance: str = "..."
    novelty: str = "should present unique insights"  # New
```

Update evaluation logic to include new criterion.

### Adding New Refinement Operations

Add method in `RefinementAgent`:
```python
def _apply_merge(self, themes, operation):
    """Merge themes using semantic similarity."""
    # Implementation
    pass
```

### Custom Output Formats

Create utility in `src/utils/`:
```python
def export_to_excel(result, output_path):
    """Export themes to Excel with multiple sheets."""
    # Implementation
    pass
```

---

## Testing Recommendations

### Unit Tests
```python
# test_generation_agent.py
def test_chunking():
    agent = GenerationAgent(api_key="test")
    transcript = "..." * 5000
    chunks = agent.chunk_transcript(transcript)
    assert len(chunks) > 0
    assert all(c.measured_size <= agent.last_chunk_plan.hard_limit for c in chunks)
```

### Integration Tests
```python
# test_integration.py
def test_full_workflow():
    tama = TAMAFramework(api_key=os.getenv("OPENAI_API_KEY"))
    result = tama.run_analysis(sample_transcript)
    assert result['accepted'] or result['refinement_iterations'] == 5
    assert len(result['final_themes']) > 0
```

### Validation Tests
```python
# test_evaluation.py
def test_evaluation_scores():
    eval_agent = EvaluationAgent(api_key="test")
    result = eval_agent.run(themes, codes)
    assert 1 <= result['average_score'] <= 5
    assert all(1 <= e['coverage_score'] <= 5 for e in result['theme_evaluations'])
```

---

## Future Enhancements

### Potential Improvements

1. **Parallel Processing**: Parallelize chunk coding
2. **Caching**: Cache LLM responses for reproducibility
3. **Batch Processing**: Process multiple transcripts
4. **Interactive Mode**: Web UI for human-in-the-loop
5. **Export Formats**: Excel, PDF, HTML outputs
6. **Advanced Analytics**: Semantic similarity metrics
7. **Multi-language**: Support non-English transcripts
8. **Fine-tuning**: Domain-specific model fine-tuning

### Scalability Considerations

**Current Limitations**:
- Sequential processing (not parallel)
- In-memory data structures
- Single-session processing

**Scaling Options**:
- Batch processing script
- Database backend for large datasets
- Distributed computing for parallel analyses
- API service deployment

---

## License

MIT License - See LICENSE file for details

---

## Support

For implementation questions:
- Review [USAGE_GUIDE.md](USAGE_GUIDE.md)
- Check [INSTALLATION.md](INSTALLATION.md)
