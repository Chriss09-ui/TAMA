"""
Generation Agent for TAMA Framework
Handles chunking, coding, and initial theme generation from interview transcripts.
"""

from bisect import bisect_right
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel
import json
from concurrent.futures import ThreadPoolExecutor

from chunking import (
    ANALYSIS_UNIT_PATTERN,
    ChunkStrategy,
    plan_chunks,
    resolve_chunk_strategy,
)
from prompts import (
    CODE_EXTRACTION_SYSTEM_PROMPT,
    THEME_GENERATION_SYSTEM_PROMPT,
    build_code_extraction_prompt,
    build_theme_generation_prompt,
)


class Chunk(BaseModel):
    """Represents a chunk of interview transcript."""
    chunk_id: int
    text: str
    start_word: int
    end_word: int
    start_char: int = 0
    end_char: int = 0
    measured_size: int = 0


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
    1. Chunking: Build dynamically sized windows at natural transcript boundaries
    2. Coding: Extract concise codes from each chunk
    3. Theme Generation: Synthesize codes into concise themes
    """

    def __init__(
        self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None,
        chunk_size: Optional[int] = None, max_workers: int = 4,
        chunk_strategy: Optional[ChunkStrategy] = None,
    ):
        """
        Initialize the Generation Agent.

        Args:
            api_key: API key for the selected model provider
            model: Model to use (default: gpt-4o)
            base_url: Optional OpenAI-compatible API endpoint
            chunk_size: Manual maximum Chinese characters or other words per chunk
            max_workers: Maximum concurrent code extraction requests
            chunk_strategy: fine, balanced, economy, or manual. Omitted values use
                manual mode when chunk_size is supplied, otherwise balanced mode.
        """
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        resolved_strategy = resolve_chunk_strategy(chunk_strategy, chunk_size)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_strategy = resolved_strategy
        self.max_workers = max_workers
        self.before_model_call = None
        self.last_chunk_plan = None

    def chunk_transcript(self, transcript: str) -> List[Chunk]:
        """
        Split a transcript using the selected sizing strategy and natural boundaries.

        Args:
            transcript: Full interview transcript text

        Returns:
            List of Chunk objects
        """
        normalized = transcript.strip()
        plan, spans = plan_chunks(
            normalized,
            strategy=self.chunk_strategy,
            manual_chunk_size=self.chunk_size,
        )
        self.last_chunk_plan = plan
        unit_ends = [match.end() for match in ANALYSIS_UNIT_PATTERN.finditer(normalized)]

        return [
            Chunk(
                chunk_id=chunk_id,
                text=normalized[span.start:span.end].strip(),
                start_word=bisect_right(unit_ends, span.start),
                end_word=bisect_right(unit_ends, span.end),
                start_char=span.start,
                end_char=span.end,
                measured_size=span.measured_size,
            )
            for chunk_id, span in enumerate(spans)
        ]

    def generate_codes_from_chunk(self, chunk: Chunk) -> List[Code]:
        """
        Generate codes from a single chunk using LLM.
        Each code should briefly describe a pattern or concept.

        Args:
            chunk: Chunk object containing transcript segment

        Returns:
            List of Code objects
        """
        prompt = build_code_extraction_prompt(chunk.text)

        if self.before_model_call:
            self.before_model_call(f"提取编码 · 片段 {chunk.chunk_id + 1}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CODE_EXTRACTION_SYSTEM_PROMPT},
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

        if len(chunks) > 1 and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
                code_batches = list(executor.map(self.generate_codes_from_chunk, chunks))
        else:
            code_batches = [self.generate_codes_from_chunk(chunk) for chunk in chunks]

        for chunk_codes in code_batches:
            # Reassign code IDs to maintain global uniqueness
            for code in chunk_codes:
                code.code_id = code_id
                all_codes.append(code)
                code_id += 1

        return all_codes

    def generate_themes(self, codes: List[Code]) -> List[Theme]:
        """
        Generate themes from codes using LLM.
        Each theme should briefly synthesize related codes.

        Args:
            codes: List of Code objects

        Returns:
            List of Theme objects
        """
        prompt = build_theme_generation_prompt(code.description for code in codes)

        if self.before_model_call:
            self.before_model_call("归纳主题")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": THEME_GENERATION_SYSTEM_PROMPT},
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
            "themes": [theme.model_dump() for theme in themes],
            "chunking": self.last_chunk_plan.to_dict() if self.last_chunk_plan else None,
        }

        # Save intermediate results if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved generation results to {save_path}")

        return result
