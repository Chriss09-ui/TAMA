# TAMA Framework - Installation and Setup Guide

## Prerequisites

- Python 3.10 or higher
- OpenAI API key (for GPT-4o access)

## Installation

### 1. Open the Project Directory

Run the following commands from the local project directory.

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Key

Set your OpenAI API key as an environment variable:

```bash
# On macOS/Linux:
export OPENAI_API_KEY='your-api-key-here'

# On Windows (Command Prompt):
set OPENAI_API_KEY=your-api-key-here

# On Windows (PowerShell):
$env:OPENAI_API_KEY='your-api-key-here'
```

Alternatively, create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Quick Start

### Run Example Analysis

```bash
python example_usage.py
```

This will:
1. Create a sample transcript (if not exists)
2. Run the complete TAMA analysis
3. Save results to `outputs/example_analysis/`

### Use Your Own Transcript

1. Place your transcript file in the `data/` directory
2. Update the `TRANSCRIPT_PATH` in `example_usage.py` or create your own script:

```python
import os
from src.tama import TAMAFramework, load_transcript

# Set API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize framework
tama = TAMAFramework(
    api_key=api_key,
    model="gpt-4o",
    max_iterations=5,
    acceptance_threshold=4.0,
    output_dir="outputs"
)

# Load and analyze transcript
transcript = load_transcript("data/your_transcript.txt")
result = tama.run_analysis(
    transcript=transcript,
    session_name="my_analysis",
    save_intermediate=True
)
```

## Project Structure

```
TAMA/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── generation_agent.py    # Generation Agent (Chunking, Coding, Themes)
│   │   ├── evaluation_agent.py    # Evaluation Agent (4 Criteria)
│   │   └── refinement_agent.py    # Refinement Agent (Add, Split, Combine, Delete)
│   ├── utils/
│   │   └── __init__.py
│   ├── __init__.py
│   └── tama.py                     # Main orchestrator
├── config/
│   └── config_template.json        # Configuration template
├── data/
│   └── sample_transcript.txt       # Sample data (auto-generated)
├── outputs/
│   └── [session_name]/             # Analysis results
│       ├── 00_final_results.json   # Complete results
│       ├── 00_summary.txt          # Human-readable summary
│       ├── 01_generation.json      # Generation output
│       ├── 02_evaluation_iter*.json# Evaluation per iteration
│       └── 03_refinement_iter*.json# Refinement per iteration
├── example_usage.py                # Example script
├── requirements.txt                # Dependencies
├── README.md                       # Main documentation
└── INSTALLATION.md                 # This file
```

## Configuration Options

You can customize the TAMA framework behavior:

```python
tama = TAMAFramework(
    api_key="your-api-key",
    model="gpt-4o",              # Model: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"
    max_iterations=5,            # Max refinement cycles (1-10)
    acceptance_threshold=4.0,    # Min score to accept themes (1.0-5.0)
    output_dir="outputs",        # Output directory
    expert_criteria={            # Optional custom evaluation criteria
        "coverage": "...",
        "actionability": "...",
        "distinctiveness": "...",
        "relevance": "..."
    }
)
```

## Custom Evaluation Criteria

You can provide domain-specific evaluation criteria from clinical experts:

```python
expert_criteria = {
    "coverage": "should comprehensively capture all important patterns and concepts from the data",
    "actionability": "should encapsulate a single concept that is clear and actionable",
    "distinctiveness": "should be clearly distinct from other themes without overlap",
    "relevance": "must accurately reflect the parents' experiences and concerns about AAOCA"
}

tama = TAMAFramework(
    api_key=api_key,
    expert_criteria=expert_criteria
)
```

## Understanding the Output

### Final Results (`00_final_results.json`)
Complete JSON output containing:
- Configuration settings
- Generation statistics
- Refinement history
- Final themes
- Evaluation scores

### Summary (`00_summary.txt`)
Human-readable summary with:
- Session metadata
- Final themes list
- Refinement history

### Intermediate Files
- `01_generation.json`: Chunks, codes, and initial themes
- `02_evaluation_iter*.json`: Evaluation results for each iteration
- `03_refinement_iter*.json`: Refinement operations for each iteration

## Troubleshooting

### API Key Issues
```
Error: Please set OPENAI_API_KEY environment variable
```
**Solution**: Set the `OPENAI_API_KEY` environment variable or create a `.env` file.

### Import Errors
```
ModuleNotFoundError: No module named 'openai'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Rate Limits
If you encounter OpenAI rate limits, consider:
- Using `gpt-4o-mini` instead of `gpt-4o`
- Adding delays between iterations
- Reducing chunk size or transcript length

## Advanced Usage

### Chunk Strategy

```python
from src.agents.generation_agent import GenerationAgent

# Automatic balanced mode is the default
generation_agent = GenerationAgent(
    api_key=api_key, model="gpt-4o", chunk_strategy="balanced"
)

# An explicit size keeps the legacy manual counting rule
manual_agent = GenerationAgent(
    api_key=api_key, model="gpt-4o", chunk_size=3000
)
```

### Programmatic Access to Results

```python
result = tama.run_analysis(transcript=transcript)

# Access final themes
for theme in result['final_themes']:
    print(f"Theme: {theme['name']}")
    print(f"Description: {theme['description']}")
    print(f"Codes: {len(theme['codes'])}")

# Access evaluation scores
print(f"Average Score: {result['metadata']['final_average_score']}")
print(f"Accepted: {result['accepted']}")
print(f"Iterations: {result['refinement_iterations']}")
```
