"""
Generation Agent for TAMA Framework
Handles chunking, coding, and initial theme generation from interview transcripts.
"""

from bisect import bisect_right
from datetime import date
from typing import List, Dict, Any, Literal, Optional
from openai import OpenAI
from pydantic import BaseModel, Field, StrictInt
import json
from concurrent.futures import ThreadPoolExecutor
import re

from chunking import (
    ANALYSIS_UNIT_PATTERN,
    ChunkStrategy,
    plan_chunks,
    resolve_chunk_strategy,
)
from code_mapping import apply_related_codes, attach_category_names, build_code_map, map_for_prompt, memos_from_map
from code_memos import code_memos_for_model, memos_from_groups
from codebook import exact_merge, semantic_merge
from prompts import (
    CODE_EXTRACTION_SYSTEM_PROMPT,
    THEME_GENERATION_SYSTEM_PROMPT,
    build_code_extraction_prompt,
    build_code_map_prompt,
    build_code_merge_prompt,
    build_theme_generation_prompt,
    build_theme_consolidation_prompt,
)
from research_profile import ResearchProfile
from research_records import SourceMetadata, apply_decisions


THEME_BATCH_SIZE = 40


def family_ids(code_id, code_by_id):
    """Return a canonical code id together with every id merged into it."""
    code = code_by_id.get(code_id)
    if code is not None and code.merged_into is not None and code.merged_into in code_by_id:
        code_id = code.merged_into
        code = code_by_id.get(code_id)
    ids = []
    if isinstance(code_id, int):
        ids.append(code_id)
    if code is not None:
        ids.extend(member for member in code.merged_from if isinstance(member, int))
    return ids


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
    code_id: StrictInt
    description: str
    source_chunks: List[int]
    excerpt: Optional[str] = None
    source_start: Optional[int] = None
    source_end: Optional[int] = None
    focus: List[str] = Field(default_factory=list)
    statement_type: str = "undetermined"
    verification_status: str = "unknown"
    open_question: str = ""
    name: str = ""
    definition: str = ""
    include: str = ""
    exclude: str = ""
    merged_from: List[StrictInt] = Field(default_factory=list)
    merged_into: Optional[StrictInt] = None
    version: int = 1
    in_vivo: bool = False
    method: str = "过程编码"
    related_code_ids: List[int] = Field(default_factory=list)
    note: str = ""
    source: SourceMetadata = Field(default_factory=SourceMetadata)
    context: str = ""
    evidence_id: str = ""
    speaker: str = ""
    definition_history: List[Dict[str, Any]] = Field(default_factory=list)
    definition_review: Literal["未复核", "需复核", "适用", "不适用"] = "未复核"
    review_note: str = ""
    definition_locked: bool = False
    separate_from: List[StrictInt] = Field(default_factory=list)


class Theme(BaseModel):
    """Represents a theme generated from codes."""
    name: str
    description: str
    codes: List[str]
    kind: str = "pattern"
    code_ids: List[int] = Field(default_factory=list)
    counterexample_code_ids: List[int] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    rationale: str = ""
    uncertain: str = ""
    category_names: List[str] = Field(default_factory=list)


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
        self.codebook_note = ""
        self.code_map = None
        self.code_memos = []
        self.code_id_start = 0
        self.source_metadata = None

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
        prompt = build_code_extraction_prompt(chunk.text, self.study, self.source_metadata)

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
            name = str(code_data.get("name") or code_data.get("description") or "").strip()
            description = str(code_data.get("description") or name).strip()
            in_vivo = code_data.get("in_vivo") in (True, "true", "True", "yes", "是")
            speaker = str(code_data.get("speaker") or "").strip()
            codes.append(Code(
                code_id=idx,
                description=description,
                name=name,
                definition="",
                include="",
                exclude="",
                in_vivo=in_vivo,
                method="实境编码" if in_vivo else "过程编码",
                source_chunks=[chunk.chunk_id],
                excerpt=verified_excerpt,
                source_start=chunk.start_char + offset if offset >= 0 else None,
                source_end=chunk.start_char + offset + len(excerpt) if offset >= 0 else None,
                focus=code_data.get("focus") or [],
                statement_type=code_data.get("statement_type") or "undetermined",
                verification_status=(code_data.get("verification_status") or "unknown") if offset >= 0 else "unknown",
                open_question=open_question,
                context=chunk.text[max(0, offset - 180):min(len(chunk.text), offset + len(excerpt) + 180)]
                    if offset >= 0 else "",
                speaker=speaker if speaker and re.search(r"(?:^|\n)\s*" + re.escape(speaker) + r"\s*[:：]", chunk.text) else "",
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
        code_id = self.code_id_start

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

    def consolidate_codebook(self, codes: List[Code]) -> List[Code]:
        """Merge identical labels, then ask the model to merge synonyms and write definitions."""
        self.codebook_note = ""
        self.code_memos = []
        exact_merge(codes)
        active = [code for code in codes if code.merged_into is None]
        if len(active) < 2:
            return codes
        payload = [
            {
                "code_id": code.code_id,
                "name": code.name or code.description,
                "description": code.description,
                "definition": code.definition,
                "include": code.include,
                "exclude": code.exclude,
                "excerpt": (code.excerpt or "")[:180],
                "source": code.source.model_dump(),
                "context": code.context,
                "separate_from": code.separate_from,
                "definition_locked": code.definition_locked,
                "evidence": self._family_evidence(code, codes),
            }
            for code in active
        ]
        if self.before_model_call:
            self.before_model_call("归并编码")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CODE_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": build_code_merge_prompt(payload, self.study)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            result = json.loads(response.choices[0].message.content)
            groups = result.get("groups") or []
            semantic_merge(codes, groups)
            self.code_memos = memos_from_groups(groups, "归并", date.today().isoformat())
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            self.codebook_note = "模型归并结果无法使用，保留各条编码；仅对同一来源、同一位置的重复证据去重。"
            print(f"  {self.codebook_note}")
        return codes

    @staticmethod
    def _family_evidence(code, codes):
        family = {code.code_id, *code.merged_from}
        return [{"code_id": item.code_id, "excerpt": item.excerpt or "", "context": item.context,
                 "source": item.source.model_dump(), "speaker": item.speaker,
                 "definition_review": item.definition_review}
                for item in codes if item.code_id in family]

    def map_codes(self, codes: List[Code]):
        """Archive a four-step code map after synonym merge and before themes."""
        active = [code for code in codes if code.merged_into is None]
        today = date.today().isoformat()
        if len(active) < 2:
            self.code_map = build_code_map(codes, None, note="规范编码少于两条，映射停在代码全集。")
            return self.code_map
        payload = [
            {
                "code_id": code.code_id,
                "name": code.name or code.description,
                "definition": code.definition,
                "include": code.include,
                "exclude": code.exclude,
                "excerpt": (code.excerpt or "")[:180],
                "source": code.source.model_dump(),
                "evidence": self._family_evidence(code, codes),
            }
            for code in active
        ]
        if self.before_model_call:
            self.before_model_call("映射编码")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CODE_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": build_code_map_prompt(payload, self.study)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            result = json.loads(response.choices[0].message.content)
            self.code_map = build_code_map(codes, result if isinstance(result, dict) else None)
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            self.code_map = build_code_map(codes, None, note="模型的代码映射无法使用，只保留了代码全集。")
            print(f"  {self.code_map['note']}")
        apply_related_codes(codes, self.code_map)
        self.code_memos.extend(memos_from_map(self.code_map, today))
        return self.code_map

    def generate_themes(self, codes: List[Code]) -> List[Theme]:
        """
        Generate themes from codes using LLM.
        Each theme should briefly synthesize related codes.

        Args:
            codes: List of Code objects

        Returns:
            List of Theme objects
        """
        code_by_id = {code.code_id: code for code in codes}
        active = [code for code in codes if code.merged_into is None]
        # A shared label is not enough to discard contextual differences.
        compact = []
        for code in active:
            compact.append({
                    "code_id": code.code_id,
                    "description": code.name or code.description,
                    "definition": code.definition,
                    "include": code.include,
                    "exclude": code.exclude,
                    "excerpt": (code.excerpt or "")[:180],
                    "focus": code.focus,
                    "verification_status": code.verification_status,
                    "merged_from": code.merged_from,
                    "duplicate_code_ids": [],
                    "source": code.source.model_dump(),
                    "evidence": self._family_evidence(code, codes),
                })
        if len(compact) <= THEME_BATCH_SIZE:
            result = self._request_themes(self._theme_prompt(compact), "归纳主题")
        else:
            candidates = []
            for start in range(0, len(compact), THEME_BATCH_SIZE):
                batch = compact[start:start + THEME_BATCH_SIZE]
                part = self._request_themes(
                    self._theme_prompt(batch),
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

        ids_by_description = {}
        for code in codes:
            ids_by_description.setdefault(code.description, []).append(code.code_id)
            if code.name:
                ids_by_description.setdefault(code.name, []).append(code.code_id)
        duplicates = {item["code_id"]: item["duplicate_code_ids"] for item in compact}
        for theme_data in result.get("themes", []):
            memo = theme_data.get("memo") if isinstance(theme_data.get("memo"), dict) else {}
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
                expanded
                for code_id in code_ids
                for expanded in (*family_ids(code_id, code_by_id), *duplicates.get(code_id, []))
            ))
            raw_counters = [
                *(theme_data.get("counterexample_code_ids") or []),
                *(theme_data.get("contradicting_code_ids") or []),
                *(memo.get("contradicting_code_ids") or []),
            ]
            counterexample_ids = [
                code_id for code_id in raw_counters
                if isinstance(code_id, int) and code_id in code_by_id
            ]
            counterexample_ids = list(dict.fromkeys(
                expanded
                for code_id in counterexample_ids
                for expanded in family_ids(code_id, code_by_id)
            ))
            kind = theme_data.get("kind") or "pattern"
            questions = list(theme_data.get("open_questions") or [])
            if not code_ids:
                kind = "evidence_gap"
                questions.append("该发现缺少可追溯的原始编码，需人工核对。")
            themes.append(Theme(
                name=theme_data["name"],
                description=theme_data["description"],
                codes=[code_by_id[code_id].description for code_id in code_ids if code_id in code_by_id],
                kind=kind,
                code_ids=code_ids,
                counterexample_code_ids=counterexample_ids,
                open_questions=questions,
                rationale=str(theme_data.get("rationale") or memo.get("rationale") or ""),
                uncertain=str(theme_data.get("uncertain") or memo.get("uncertain") or ""),
            ))

        payload = [theme.model_dump() for theme in themes]
        attach_category_names(payload, [code.model_dump() for code in codes], self.code_map)
        for theme, item in zip(themes, payload):
            theme.category_names = list(item.get("category_names") or [])
        return themes

    def _theme_prompt(self, codes) -> str:
        return build_theme_generation_prompt(
            codes,
            self.study,
            map_for_prompt(self.code_map),
            code_memos_for_model(self.code_memos),
        )

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

    def run(self, transcript: str, save_path: str = None, previous_codes=None,
            source_metadata=None, confirmed_decisions=None) -> Dict[str, Any]:
        """
        Run the complete generation pipeline: chunking -> coding -> theme generation.

        Args:
            transcript: Full interview transcript text
            save_path: Optional path to save intermediate results

        Returns:
            Dictionary containing chunks, codes, and themes
        """
        print("Step 1/5: Chunking transcript...")
        chunks = self.chunk_transcript(transcript)
        print(f"  Generated {len(chunks)} chunks")

        print("Step 2/5: Generating codes from chunks...")
        source = SourceMetadata.model_validate(source_metadata or {})
        if not source.source_id:
            from uuid import uuid4
            source.source_id = uuid4().hex
        self.source_metadata = source.model_dump()
        prior = [Code.model_validate(item) for item in previous_codes or []]
        self.code_id_start = max((item.code_id for item in prior), default=-1) + 1
        codes = self.generate_codes(chunks)
        for code in codes:
            code.source = source.model_copy(deep=True)
            code.evidence_id = f"{source.source_id}:{code.code_id}"
        codes = prior + codes
        apply_decisions(codes, confirmed_decisions)
        print(f"  Generated {len(codes)} codes")

        print("Step 3/5: Consolidating the codebook...")
        codes = self.consolidate_codebook(codes)
        active_codes = sum(1 for code in codes if code.merged_into is None)
        print(f"  Codebook entries: {active_codes}")

        print("Step 4/5: Mapping codes into categories...")
        code_map = self.map_codes(codes)
        print(f"  Code-map steps: {len(code_map.get('iterations') or [])}")

        print("Step 5/5: Generating themes from codes...")
        themes = self.generate_themes(codes)
        print(f"  Generated {len(themes)} themes")

        result = {
            "chunks": [chunk.model_dump() for chunk in chunks],
            "codes": [code.model_dump() for code in codes],
            "themes": [theme.model_dump() for theme in themes],
            "analytic_storyline": self.analytic_storyline,
            "codebook_note": self.codebook_note,
            "code_map": self.code_map,
            "code_memos": self.code_memos,
            "chunking": self.last_chunk_plan.to_dict() if self.last_chunk_plan else None,
        }

        # Save intermediate results if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved generation results to {save_path}")

        return result
