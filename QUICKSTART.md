# Threadline Framework - Quick Start Guide

Get started with Threadline in 5 minutes!

## Step 1: Install (1 minute)

```bash
# Run from the project directory
pip install -r requirements.txt
```

## Step 2: Set API Key (30 seconds)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

## Step 3: Run Example (3 minutes)

```bash
# Run the example analysis
python example_usage.py
```

This will:
1. Create a sample interview transcript
2. Run one independent analysis, including one full-text review by default
3. Save results to `outputs/example_analysis/`

## Step 4: View Results

Check your results:

```bash
# View human-readable summary
cat outputs/example_analysis/00_summary.txt

# View final themes (JSON)
cat outputs/example_analysis/00_final_results.json
```

## Step 5: Analyze Your Own Data

Create a Python script:

```python
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from tama import ThreadlineFramework, load_transcript

# Initialize Threadline
framework = ThreadlineFramework(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    max_iterations=5,
    acceptance_threshold=4.0
)

# Load your transcript
transcript = load_transcript("path/to/your/transcript.txt")

# Run analysis
result = framework.run_analysis(
    transcript=transcript,
    session_name="my_first_analysis",
    save_intermediate=True
)

# Print results
print(f"\n✓ Analysis complete!")
print(f"  Themes generated: {len(result['final_themes'])}")
score = result["metadata"]["final_average_score"]
print(f"  Quality score: {score if score is not None else 'unscored'}")
print(f"  Results saved to: outputs/my_first_analysis/")
```

## What You'll Get

After running Threadline, you'll find in `outputs/[session_name]/`:

📄 **00_summary.txt** - Human-readable summary of themes
```
1. Seeking clarity through multiple medical opinions
   Parents actively pursue multiple medical opinions to gain reassurance...
   Associated codes: 6

2. Emotional impact of diagnosis uncertainty
   The unexpected diagnosis creates anxiety and fear about child's safety...
   Associated codes: 8
```

📊 **00_final_results.json** - Complete analysis data
- Submitted original text and its SHA-256 checksum
- Full-text review status and findings
- All themes with descriptions
- Evaluation scores
- Refinement history
- Metadata

🔍 **Intermediate Files** - For transparency
- `01_generation.json` - Initial chunks, codes, themes
- `02_evaluation_iter*.json` - Quality evaluations
- `03_refinement_iter*.json` - Refinement operations

## Common Customizations

### Use Faster Model (Lower Cost)
```python
framework = ThreadlineFramework(
    api_key=api_key,
    model="gpt-4o-mini"  # Faster and cheaper
)
```

### Higher Quality Standards
```python
framework = ThreadlineFramework(
    api_key=api_key,
    acceptance_threshold=4.5,  # Stricter quality requirement
    max_iterations=7           # More refinement opportunities
)
```

### Custom Evaluation Criteria
```python
expert_criteria = {
    "coverage": "should capture all patient concerns and experiences",
    "actionability": "should represent a single, clear medical concept",
    "distinctiveness": "should be clearly different from other themes",
    "relevance": "must reflect actual patient statements"
}

framework = ThreadlineFramework(
    api_key=api_key,
    expert_criteria=expert_criteria
)
```

## Understanding the Output

### Theme Structure
Each theme contains:
- **Name**: Short, descriptive title
- **Description**: ~25 words explaining the theme
- **Codes**: List of related codes from the data

### Quality Scores
Themes are rated 1-5 on four criteria:
- **Coverage**: Comprehensiveness
- **Actionability**: Single, clear concept
- **Distinctiveness**: Uniqueness
- **Relevance**: Accuracy to data

Scores assess candidate themes, not research validity or saturation. No valid evidence produces an unscored result. Partial or failed full-text review cannot pass the overall analysis.

## Next Steps

- **Read full documentation**: [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Detailed installation**: [INSTALLATION.md](INSTALLATION.md)
- **Understand the framework**: [README.md](README.md)

## Troubleshooting

**Can't find module 'openai'?**
→ Run `pip install -r requirements.txt`

**API key error?**
→ Set environment variable: `export OPENAI_API_KEY='your-key'`

**Low quality themes?**
→ Try increasing `max_iterations` or providing custom `expert_criteria`

**Too many themes?**
→ Increase `acceptance_threshold` to force more consolidation

For more help, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

---

**Ready to start?** Run `python example_usage.py` and explore your first thematic analysis!
