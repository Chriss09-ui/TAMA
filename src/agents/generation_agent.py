"""
Generation Agent for TAMA Framework
Handles chunking, coding, and initial theme generation from interview transcripts.
"""

from bisect import bisect_right
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
import json
from concurrent.futures import ThreadPoolExecutor
import re

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
    build_theme_consolidation_prompt,
)
from research_profile import ResearchProfile


THEME_BATCH_SIZE = 40


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
    excerpt: Optional[str] = None
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    focus: List[str] = Field(default_factory=list)
    statement_type: str = "undetermined"
    verification_status: str = "unknown"
    open_question: str = ""


class Theme(BaseModel):
    """Represents a theme generated from codes."""
    name: str
    description: str
    codes: List[str]
    kind: str = "pattern"
    code_ids: List[int] = Field(default_factory=list)
    counterexample_code_ids: List[int] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)


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
        study: Optional[ResearchProfile] = None,
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
        self.study = study or ResearchProfile()
        self.analytic_storyline = ""

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

        chunks = []
        for chunk_id, span in enumerate(spans):
            raw_text = normalized[span.start:span.end]
            leading_space = len(raw_text) - len(raw_text.lstrip())
            trailing_space = len(raw_text) - len(raw_text.rstrip())
            start_char = span.start + leading_space
            end_char = span.end - trailing_space
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=normalized[start_char:end_char],
                start_word=bisect_right(unit_ends, start_char),
                end_word=bisect_right(unit_ends, end_char),
                start_char=start_char,
                end_char=end_char,
                measured_size=span.measured_size,
            ))
        return chunks

    def generate_codes_from_chunk(self, chunk: Chunk) -> List[Code]:
        """
        Generate codes from a single chunk using LLM.
        Each code should briefly describe a pattern or concept.

        Args:
            chunk: Chunk object containing transcript segment

        Returns:
            List of Code objects
        """
        prompt = build_code_extraction_prompt(chunk.text, self.study)

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
            excerpt = code_data.get("excerpt")
            offset = chunk.text.find(excerpt) if isinstance(excerpt, str) and excerpt else -1
            verified_excerpt = excerpt if offset >= 0 else None
            open_question = str(code_data.get("open_question") or "")
            if excerpt and offset < 0:
                open_question = f"{open_question}；原文摘录未匹配到输入片段，需人工核对".strip("；")
            elif not excerpt:
                open_question = f"{open_question}；缺少逐字原文，需人工核对".strip("；")
            codes.append(Code(
                code_id=idx,
                description=code_data["description"],
                source_chunks=[chunk.chunk_id],
                excerpt=verified_excerpt,
                source_start=chunk.start_char + offset if offset >= 0 else None,
                source_end=chunk.start_char + offset + len(excerpt) if offset >= 0 else None,
                focus=code_data.get("focus") or [],
                statement_type=code_data.get("statement_type") or "undetermined",
                verification_status=(code_data.get("verification_status") or "unknown") if offset >= 0 else "unknown",
                open_question=open_question,
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
        # Collapse identical labels for synthesis while retaining every raw code.
        canonical = {}
        for code in codes:
            key = re.sub(r"\s+", "", code.description).casefold()
            if key not in canonical:
                canonical[key] = {
                    "code_id": code.code_id, "description": code.description,
                    "excerpt": (code.excerpt or "")[:180], "focus": code.focus,
                    "verification_status": code.verification_status,
                    "duplicate_code_ids": [],
                }
            else:
                canonical[key]["duplicate_code_ids"].append(code.code_id)
        compact = list(canonical.values())
        if len(compact) <= THEME_BATCH_SIZE:
            result = self._request_themes(build_theme_generation_prompt(compact, self.study), "归纳主题")
        else:
            candidates = []
            for start in range(0, len(compact), THEME_BATCH_SIZE):
                batch = compact[start:start + THEME_BATCH_SIZE]
                part = self._request_themes(
                    build_theme_generation_prompt(batch, self.study),
                    f"归纳主题 · 第 {start // THEME_BATCH_SIZE + 1} 组",
                )
                candidates.extend(part.get("themes", []))
            while len(candidates) > THEME_BATCH_SIZE:
                grouped = []
                for start in range(0, len(candidates), THEME_BATCH_SIZE):
                    part = self._request_themes(
                        build_theme_consolidation_prompt(candidates[start:start + THEME_BATCH_SIZE], self.study),
                        "合并候选主题",
                    )
                    grouped.extend(part.get("themes", []))
                if len(grouped) >= len(candidates):
                    raise ValueError("候选主题未能归并，请检查模型返回或增大主题分批上限。")
                candidates = grouped
            result = self._request_themes(
                build_theme_consolidation_prompt(candidates, self.study), "合并候选主题",
            )
        self.analytic_storyline = result.get("analytic_storyline") or ""
        themes = []

        code_by_id = {code.code_id: code for code in codes}
        ids_by_description = {}
        for code in codes:
            ids_by_description.setdefault(code.description, []).append(code.code_id)
        duplicates = {item["code_id"]: item["duplicate_code_ids"] for item in compact}
        for theme_data in result.get("themes", []):
            code_ids = [
                code_id for code_id in theme_data.get("code_ids", [])
                if isinstance(code_id, int) and code_id in code_by_id
            ]
            if not code_ids:
                code_ids = [
                    code_id
                    for description in theme_data.get("codes", [])
                    for code_id in ids_by_description.get(description, [])
                ]
            code_ids = list(dict.fromkeys(
                expanded for code_id in code_ids for expanded in (code_id, *duplicates.get(code_id, []))
            ))
            counterexample_ids = [
                code_id for code_id in theme_data.get("counterexample_code_ids", [])
                if isinstance(code_id, int) and code_id in code_by_id
            ]
            kind = theme_data.get("kind") or "pattern"
            questions = list(theme_data.get("open_questions") or [])
            if not code_ids:
                kind = "evidence_gap"
                questions.append("该发现缺少可追溯的原始编码，需人工核对。")
            themes.append(Theme(
                name=theme_data["name"],
                description=theme_data["description"],
                codes=[code_by_id[code_id].description for code_id in code_ids],
                kind=kind,
                code_ids=code_ids,
                counterexample_code_ids=list(dict.fromkeys(counterexample_ids)),
                open_questions=questions,
            ))

        return themes

    def _request_themes(self, prompt: str, stage: str) -> Dict[str, Any]:
        if self.before_model_call:
            self.before_model_call(stage)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": THEME_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

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
            "analytic_storyline": self.analytic_storyline,
            "chunking": self.last_chunk_plan.to_dict() if self.last_chunk_plan else None,
        }

        # Save intermediate results if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved generation results to {save_path}")

        return result
