"""
Example Usage of Threadline Framework
Demonstrates how to run thematic analysis on interview transcripts.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tama import ThreadlineFramework, load_transcript
from decisions.jev_client import JevDecisionClient
from model_config import DEFAULT_MIMO_MODEL, api_model_name
from api_settings import get_api_setting


def get_decision_provider():
    """Prefer the Jev decision API when JEV_API_KEY is set; otherwise the main model decides."""
    jev_api_key = get_api_setting("JEV_API_KEY", "").strip()
    if jev_api_key:
        return JevDecisionClient(
            api_key=jev_api_key,
            base_url=get_api_setting("JEV_BASE_URL") or None,
            model=get_api_setting("JEV_MODEL") or None,
        )
    return None


def get_model_config():
    """Select MiMo, DeepSeek, or a custom provider from the configured API keys."""
    mimo_api_key = get_api_setting("MIMO_API_KEY")
    if mimo_api_key:
        return (
            mimo_api_key,
            get_api_setting("MIMO_MODEL", DEFAULT_MIMO_MODEL),
            get_api_setting("MIMO_BASE_URL") or "https://api.xiaomimimo.com/v1"
        )

    deepseek_api_key = get_api_setting("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        return (
            deepseek_api_key,
            api_model_name("DeepSeek", get_api_setting("DEEPSEEK_MODEL", "deepseek-flash")),
            get_api_setting("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        )

    custom_api_key = get_api_setting("CUSTOM_API_KEY", "").strip()
    if not custom_api_key:
        raise ValueError("Please set MIMO_API_KEY, DEEPSEEK_API_KEY, or CUSTOM_API_KEY environment variable")
    custom_model = get_api_setting("CUSTOM_MODEL", "").strip()
    custom_base_url = get_api_setting("CUSTOM_BASE_URL", "").strip()
    if not custom_model:
        raise ValueError("Please set CUSTOM_MODEL environment variable")
    if not custom_base_url:
        raise ValueError("Please set CUSTOM_BASE_URL environment variable")
    return custom_api_key, custom_model, custom_base_url


def main():
    """
    Main function demonstrating Threadline framework usage.
    """
    # Configuration
    api_key, model, base_url = get_model_config()

    # Path to your transcript file
    TRANSCRIPT_PATH = "data/sample_transcript.txt"

    # Optional: Define study-specific evaluation criteria
    # If not provided, default criteria will be used
    expert_criteria = {
        "coverage": "主题应覆盖访谈材料中的重要规律。",
        "actionability": "每个主题应表达一个清楚、具体且便于理解的概念。",
        "distinctiveness": "各主题之间应有明确区分，避免重复或重叠。",
        "relevance": "每个主题都应有访谈材料作为依据，不添加未经证实的假设。",
    }

    # Initialize Threadline framework
    print("Initializing Threadline Framework...")
    framework = ThreadlineFramework(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_iterations=5,  # Maximum refinement iterations
        acceptance_threshold=4.0,  # Minimum score (out of 5) to accept themes
        output_dir="outputs",  # Directory to save results
        expert_criteria=expert_criteria,  # Optional custom criteria
        decision_provider=get_decision_provider(),  # Optional Jev decision provider
    )

    # Check if transcript exists
    if not os.path.exists(TRANSCRIPT_PATH):
        print(f"\nError: Transcript file not found at {TRANSCRIPT_PATH}")
        print("\nCreating sample transcript for demonstration...")
        create_sample_transcript(TRANSCRIPT_PATH)

    # Load transcript
    print(f"\nLoading transcript from {TRANSCRIPT_PATH}...")
    transcript = load_transcript(TRANSCRIPT_PATH)
    print(f"Transcript loaded: {len(transcript.split())} words")

    # Run Threadline analysis
    print("\nStarting Threadline analysis...\n")
    result = framework.run_analysis(
        transcript=transcript,
        session_name="example_analysis",  # Optional: name for this session
        save_intermediate=True  # Save intermediate results for transparency
    )

    # Display summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nFinal Themes ({len(result['final_themes'])}):")
    for idx, theme in enumerate(result['final_themes'], 1):
        print(f"\n{idx}. {theme['name']}")
        print(f"   {theme['description']}")

    print(f"\nSession files saved in: outputs/{result['session_name']}/")
    print("  - 00_final_results.json: Complete analysis results")
    print("  - 00_summary.txt: Human-readable summary")
    print("  - 01_generation.json: Initial generation output")
    print("  - 02_evaluation_iter*.json: Evaluation results per iteration")
    print("  - 03_refinement_iter*.json: Refinement results per iteration")


def create_sample_transcript(output_path: str):
    """
    Create a sample transcript for demonstration purposes.

    Args:
        output_path: Path to save sample transcript
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sample_text = """
Interview with Parent A (Mother, Age 42):

Interviewer: Can you tell me about when you first learned about your child's diagnosis?

Parent A: It was completely unexpected. We went in for a routine sports physical, and the doctor heard a murmur.
They referred us to a cardiologist, and that's when we found out about the AAOCA. I had never heard of it before.
The uncertainty was the worst part - not knowing if my child could play sports, not knowing if they were at risk
for sudden cardiac death. I spent countless hours searching online, trying to find information, but there was so
little available. I felt completely lost and scared.

Interviewer: How did you make decisions about treatment?

Parent A: It was incredibly difficult. We saw three different cardiologists because we wanted to make sure we
were getting the right advice. Each one had slightly different recommendations, which made it even more confusing.
We ultimately decided on surgery because we couldn't live with the constant fear. The waiting period before the
surgery was agonizing - every time my child complained about anything, I worried it could be cardiac-related.
I became hypervigilant, constantly watching for symptoms.

Interview with Parent B (Father, Age 45):

Interviewer: What was your experience with the medical system?

Parent B: The lack of data was frustrating. When we asked about outcomes, success rates, long-term prognosis,
the doctors couldn't give us definitive answers because AAOCA is so rare. They were doing their best, but
I'm an engineer - I like numbers, statistics, concrete information. Not having that made the decision-making
process incredibly stressful. We also struggled with insurance approval and understanding what would be covered.

Interviewer: How did this affect your family life?

Parent B: It completely changed our daily routine. We had to restrict our child's activities, which was hard
because they were very athletic. They didn't understand why they couldn't play with their friends the same way.
There was a lot of emotional stress - anxiety, fear, uncertainty. We tried to stay positive for our child,
but internally we were terrified. My spouse and I had many sleepless nights, and it put a strain on our
relationship at times because we handled the stress differently.

Interview with Parent C (Mother, Age 38):

Interviewer: What support did you receive?

Parent C: Initially, we felt very isolated because none of our friends or family had heard of AAOCA.
The medical team was supportive, but I needed to talk to other parents who understood what we were going through.
I eventually found an online support group, which was invaluable. Hearing from other parents who had been through
this gave me hope and practical advice. I wish there had been more organized support systems in place.

Interviewer: What would have helped during this time?

Parent C: Better communication from the healthcare providers would have helped enormously. Sometimes medical
terminology was used that we didn't understand, and we were too overwhelmed to ask for clarification in the moment.
Having written materials to take home would have been useful. Also, more information about what to expect after
surgery - the recovery process, potential complications, when to worry. We were sent home with basic instructions
but had so many questions afterward. A dedicated care coordinator or nurse we could contact would have reduced
our anxiety significantly.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_text.strip())

    print(f"Sample transcript created at: {output_path}")


if __name__ == "__main__":
    main()
