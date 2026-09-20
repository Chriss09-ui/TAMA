"""
Generation Agent for TAMA Framework
Handles chunking, coding, and initial theme generation from interview transcripts.
"""

from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel
import json


class Chunk(BaseModel):
    """Represents a chunk of interview transcript."""
    chunk_id: int
    text: str
    start_word: int
    end_word: int


class Code(BaseModel):
    """Represents a code extracted from chunks."""
    code_id: int
    description: str
    source_chunks: List[int]


class Theme(BaseModel):
    """Represents a theme generated from codes."""
    name: str
    description: str
    codes: List[str]


class GenerationAgent:
    """
    Generation Agent that processes interview transcripts through:
    1. Chunking: Split transcripts into manageable segments (3-5k words per chunk)
    2. Coding: Extract codes from each chunk (< 25 words per code)
    3. Theme Generation: Synthesize codes into themes (25 words per theme)
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the Generation Agent.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chunk_size = 4000  # words per chunk (3-5k as per diagram)

    def chunk_transcript(self, transcript: str) -> List[Chunk]:
        """
        Split interview transcript into chunks of 3-5k words.

        Args:
            transcript: Full interview transcript text

        Returns:
            List of Chunk objects
        """
        words = transcript.split()
        chunks = []
        chunk_id = 0

        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_word=i,
                end_word=min(i + self.chunk_size, len(words))
            ))
            chunk_id += 1

        return chunks

    def generate_codes_from_chunk(self, chunk: Chunk) -> List[Code]:
        """
        Generate codes from a single chunk using LLM.
        Each code should be < 25 words describing a pattern or concept.

        Args:
            chunk: Chunk object containing transcript segment

        Returns:
            List of Code objects
        """
        prompt = f"""You are a qualitative research expert conducting thematic analysis of clinical interview transcripts.

Your task is to extract codes from the following interview transcript chunk.

INSTRUCTIONS:
- Identify key patterns, concepts, and ideas expressed in the text
- Each code should be a concise description (< 25 words)
- Codes should capture meaningful units of information
- Focus on patient/parent experiences, emotions, concerns, and perspectives
- Output your codes as a JSON array of objects with "description" field

CHUNK TEXT:
{chunk.text}

Provide your response as a JSON object with this structure:
{{
  "codes": [
    {{"description": "Description of code 1"}},
    {{"description": "Description of code 2"}},
    ...
  ]
}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a qualitative research expert specializing in thematic analysis of clinical interviews."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        codes = []

        for idx, code_data in enumerate(result.get("codes", [])):
            codes.append(Code(
                code_id=idx,
                description=code_data["description"],
                source_chunks=[chunk.chunk_id]
            ))

        return codes

    def generate_codes(self, chunks: List[Chunk]) -> List[Code]:
        """
        Generate codes from all chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            List of all Code objects from all chunks
        """
        all_codes = []
        code_id = 0

        for chunk in chunks:
            chunk_codes = self.generate_codes_from_chunk(chunk)

            # Reassign code IDs to maintain global uniqueness
            for code in chunk_codes:
                code.code_id = code_id
                all_codes.append(code)
                code_id += 1

        return all_codes

    def generate_themes(self, codes: List[Code]) -> List[Theme]:
        """
        Generate themes from codes using LLM.
        Each theme should be ~25 words and synthesize related codes.

        Args:
            codes: List of Code objects

        Returns:
            List of Theme objects
        """
        codes_text = "\n".join([f"- {code.description}" for code in codes])

        prompt = f"""You are a qualitative research expert conducting thematic analysis of clinical interview transcripts.

Your task is to synthesize the following codes into coherent themes.

INSTRUCTIONS:
- Group related codes into broader themes
- Each theme should have a clear, descriptive name
- Each theme description should be approximately 25 words
- Themes should capture meaningful patterns across the data
- Ensure themes are distinct from each other
- Each theme should reference the specific codes it encompasses

CODES:
{codes_text}

Provide your response as a JSON object with this structure:
{{
  "themes": [
    {{
      "name": "Theme name",
      "description": "Approximately 25-word description of the theme",
      "codes": ["List of code descriptions that belong to this theme"]
    }},
    ...
  ]
}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a qualitative research expert specializing in thematic analysis of clinical interviews."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        themes = []

        for theme_data in result.get("themes", []):
            themes.append(Theme(
                name=theme_data["name"],
                description=theme_data["description"],
                codes=theme_data["codes"]
            ))

        return themes

    def run(self, transcript: str, save_path: str = None) -> Dict[str, Any]:
        """
        Run the complete generation pipeline: chunking -> coding -> theme generation.

        Args:
            transcript: Full interview transcript text
            save_path: Optional path to save intermediate results

        Returns:
            Dictionary containing chunks, codes, and themes
        """
        print("Step 1/3: Chunking transcript...")
        chunks = self.chunk_transcript(transcript)
        print(f"  Generated {len(chunks)} chunks")

        print("Step 2/3: Generating codes from chunks...")
        codes = self.generate_codes(chunks)
        print(f"  Generated {len(codes)} codes")

        print("Step 3/3: Generating themes from codes...")
        themes = self.generate_themes(codes)
        print(f"  Generated {len(themes)} themes")

        result = {
            "chunks": [chunk.model_dump() for chunk in chunks],
            "codes": [code.model_dump() for code in codes],
            "themes": [theme.model_dump() for theme in themes]
        }

        # Save intermediate results if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved generation results to {save_path}")

        return result
