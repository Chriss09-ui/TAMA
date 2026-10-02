# Threadline Project Structure

```
threadline/
│
├── README.md                        # Main documentation
├── QUICKSTART.md                    # 5-minute quick start guide
├── INSTALLATION.md                  # Detailed installation instructions
├── USAGE_GUIDE.md                   # Comprehensive usage guide
├── PROJECT_STRUCTURE.md             # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
│
├── example_usage.py                 # Example script demonstrating framework usage
│
├── figure/                          # Documentation images
│   └── TAMA-Workflow.png           # Framework workflow diagram
│
├── config/                          # Configuration templates
│   └── config_template.json        # Template for custom configurations
│
├── data/                            # Input data directory
│   └── sample_transcript.txt       # Auto-generated sample (created on first run)
│
├── src/                             # Source code
│   ├── __init__.py                 # Package initialization
│   ├── tama.py                     # Main Threadline orchestrator
│   ├── chunking.py                 # Dynamic sizing and natural-boundary splitting
│   │
│   ├── agents/                     # Multi-agent components
│   │   ├── __init__.py
│   │   ├── generation_agent.py    # Generation Agent (Chunk → Code → Theme)
│   │   ├── evaluation_agent.py    # Evaluation Agent (4 criteria, hybrid scoring)
│   │   └── refinement_agent.py    # Refinement Agent (Add/Split/Combine/Delete)
│   │
│   ├── decisions/                  # Decision provider layer (typed probabilistic decisions)
│   │   ├── __init__.py
│   │   ├── base.py                # DecisionProvider ABC, error type, state type
│   │   ├── models.py              # DecisionQuestion / DecisionAnswer
│   │   ├── llm_client.py          # JSON-mode LLM client (default, works with all providers)
│   │   └── jev_client.py          # TypeSafe Jev System One HTTP client
│   │
│   ├── prompts.py                  # All model prompts, rubrics, and connection-test questions
│   │
│   └── utils/                      # Utility functions
│       └── __init__.py
│
└── outputs/                         # Analysis results (auto-created)
    └── [session_name]/             # Results for each analysis session
        ├── 00_final_results.json   # Complete analysis results (JSON)
        ├── 00_summary.txt          # Human-readable summary
        ├── 01_generation.json      # Generation phase output
        ├── 02_evaluation_iter1.json # Evaluation results (iteration 1)
        ├── 02_evaluation_iter2.json # Evaluation results (iteration 2)
        ├── 03_refinement_iter1.json # Refinement results (iteration 1)
        └── 03_refinement_iter2.json # Refinement results (iteration 2)
```

## Key Files Explained

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Overview, features, basic usage |
| `QUICKSTART.md` | Get started in 5 minutes |
| `INSTALLATION.md` | Detailed setup instructions |
| `USAGE_GUIDE.md` | Comprehensive usage documentation |
| `PROJECT_STRUCTURE.md` | This file - project organization |

### Source Code

| File | Description |
|------|-------------|
| `src/tama.py` | Main orchestrator coordinating all agents |
| `src/chunking.py` | Shared automatic/manual chunk planner used by the UI and Generation Agent |
| `src/agents/generation_agent.py` | Chunks transcripts, extracts codes, generates themes |
| `src/agents/evaluation_agent.py` | Evaluates themes on 4 criteria, provides feedback |
| `src/agents/refinement_agent.py` | Refines themes using Add/Split/Combine/Delete |
| `src/decisions/` | Decision provider layer: typed scores/classifications with confidence |

### Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | Python package dependencies |
| `config/config_template.json` | Template for custom settings |
| `.gitignore` | Files to exclude from version control |

### Examples & Data

| File | Purpose |
|------|---------|
| `example_usage.py` | Runnable example demonstrating the framework |
| `data/` | Directory for input transcripts |
| `outputs/` | Directory for analysis results (auto-created) |

## Workflow Through Files

1. **User starts**: Runs `example_usage.py` or custom script
2. **Initialization**: `src/tama.py` creates ThreadlineFramework instance
3. **Generation**: `generation_agent.py` processes transcript
   - `chunking.py` resolves the selected strategy and preserves natural boundaries
   - Output: `outputs/[session]/01_generation.json`
4. **Iteration Loop**:
   - **Evaluation**: `evaluation_agent.py` assesses themes
     - Output: `outputs/[session]/02_evaluation_iterN.json`
   - **Refinement**: `refinement_agent.py` improves themes
     - Output: `outputs/[session]/03_refinement_iterN.json`
5. **Completion**: Final results saved
   - `outputs/[session]/00_final_results.json`
   - `outputs/[session]/00_summary.txt`

## Output File Naming Convention

| Prefix | Meaning | Example |
|--------|---------|---------|
| `00_` | Final/summary files | `00_final_results.json` |
| `01_` | Generation phase | `01_generation.json` |
| `02_` | Evaluation phase | `02_evaluation_iter1.json` |
| `03_` | Refinement phase | `03_refinement_iter1.json` |

The numbering ensures proper ordering when files are sorted alphabetically.

## Adding Custom Components

### Custom Agent
1. Create new file in `src/agents/`
2. Implement agent logic
3. Import in `src/agents/__init__.py`
4. Integrate into `src/tama.py` workflow

### Custom Utility
1. Create function in `src/utils/`
2. Import where needed

### Custom Configuration
1. Copy `config/config_template.json`
2. Modify parameters
3. Load in your script:
```python
import json
with open('config/my_config.json') as f:
    config = json.load(f)

framework = ThreadlineFramework(
    api_key=api_key,
    model=config['model'],
    max_iterations=config['max_iterations'],
    acceptance_threshold=config['acceptance_threshold'],
    expert_criteria=config['evaluation_criteria']
)
```

## File Size Considerations

Typical file sizes for a standard interview analysis:

| File | Typical Size |
|------|-------------|
| Input transcript | 10-100 KB |
| `01_generation.json` | 50-200 KB |
| `02_evaluation_iterN.json` | 20-50 KB |
| `03_refinement_iterN.json` | 30-80 KB |
| `00_final_results.json` | 100-300 KB |
| `00_summary.txt` | 5-15 KB |

Total per analysis session: ~300-800 KB

## Version Control

Files tracked in Git:
- All source code (`src/`)
- Documentation (`.md` files)
- Configuration templates (`config/`)
- Example scripts (`.py` files)
- Requirements (`requirements.txt`)

Files NOT tracked in Git (per `.gitignore`):
- Output files (`outputs/`)
- API keys (`.env`)
- Python cache (`__pycache__/`)
- Virtual environments (`venv/`)
- IDE settings (`.vscode/`, `.idea/`)

## Development Setup

For contributors:

```bash
# Run from the project directory
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install jupyter pytest black flake8

# Set up API key
export OPENAI_API_KEY='your-key'

# Run tests (if available)
python -m pytest

# Format code
black src/

# Run example
python example_usage.py
```

## Extending Threadline

### Add New Evaluation Criterion

Edit `src/agents/evaluation_agent.py`:

```python
class EvaluationCriteria(BaseModel):
    coverage: str = "..."
    actionability: str = "..."
    distinctiveness: str = "..."
    relevance: str = "..."
    novelty: str = "should present unique insights"  # New criterion
```

### Add New Refinement Operation

Edit `src/agents/refinement_agent.py`:

```python
def _apply_merge(self, themes, operation):
    """Merge similar themes into a unified theme."""
    # Implementation
    pass
```

### Custom Output Format

Create new utility in `src/utils/export_utils.py`:

```python
def export_to_csv(themes, output_path):
    """Export themes to CSV format."""
    import csv
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Description', 'Codes'])
        for theme in themes:
            writer.writerow([
                theme['name'],
                theme['description'],
                '; '.join(theme['codes'])
            ])
```

---

For questions about project structure, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
