"""
Example Usage of TAMA Framework
Demonstrates how to run thematic analysis on clinical interview transcripts.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tama import TAMAFramework, load_transcript


def main():
    """
    Main function demonstrating TAMA framework usage.
    """
    # Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Path to your transcript file
    TRANSCRIPT_PATH = "data/sample_transcript.txt"

    # Optional: Define custom evaluation criteria from cardiac expert
    # If not provided, default criteria will be used
    expert_criteria = {
    "coverage": (
        "The generated themes should comprehensively capture the key aspects of parents’ "
        "lived experiences while caring for children with AAOCA from the transcripts."
    ),
    "actionability": (
        "Each theme should encapsulate a single concept that provides clear, specific, "
        "and meaningful insights. These insights should be actionable and useful for "
        "informing interventions, resources, or research."
    ),
    "distinctiveness": (
        "Each theme should be clearly distinct from one another, with no overlaps or redundancies."
    ),
    "relevance": (
        "Each theme should clearly reflect the parents’ lived experiences, concerns, and needs, "
        "without confusing or overlapping with themes related to the child/patient’s feelings, "
        "concerns, or experiences."
    ),
}

    # Initialize TAMA framework
    print("Initializing TAMA Framework...")
    tama = TAMAFramework(
        api_key=API_KEY,
        model="gpt-4o",  # or "gpt-4o-mini" for faster/cheaper processing
        max_iterations=5,  # Maximum refinement iterations
        acceptance_threshold=4.0,  # Minimum score (out of 5) to accept themes
        output_dir="outputs",  # Directory to save results
        expert_criteria=expert_criteria  # Optional custom criteria
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

    # Run TAMA analysis
    print("\nStarting TAMA analysis...\n")
    result = tama.run_analysis(
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
