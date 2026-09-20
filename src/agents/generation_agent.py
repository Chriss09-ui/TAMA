"""
Generation Agent for TAMA Framework
Handles chunking, coding, and initial theme generation from interview transcripts.
"""

from typing import List, Dict, Any, Optional
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
    2. Coding: Extract concise codes from each chunk
    3. Theme Generation: Synthesize codes into concise themes
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None):
        """
        Initialize the Generation Agent.

        Args:
            api_key: API key for the selected model provider
            model: Model to use (default: gpt-4o)
            base_url: Optional OpenAI-compatible API endpoint
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.chunk_size = 4000  # words per chunk (3-5k as per diagram)
        self.before_model_call = None

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
        Each code should briefly describe a pattern or concept.

        Args:
            chunk: Chunk object containing transcript segment

        Returns:
            List of Code objects
        """
        prompt = f"""你是一名质性研究者，正在对访谈文本进行归纳式主题分析。

请从以下访谈片段中提取编码。

要求：
- 识别文本中有意义的经历、行为、观点和规律
- 每条编码用简短词组或短句表达，并对应一段有意义的信息
- 不要预设文本未说明的人群、场景或研究主题
- 编码内容使用与访谈文本相同的语言，不编造文本中没有的信息
- 返回包含 "codes" 数组的 JSON 对象，每条编码包含 "description" 字段

访谈片段：
{chunk.text}

只返回符合以下结构的 JSON 对象，不添加解释或 Markdown：
{{
  "codes": [
    {{"description": "第一条编码的简短描述"}},
    {{"description": "第二条编码的简短描述"}}
  ]
}}
"""

        if self.before_model_call:
            self.before_model_call(f"提取编码 · 片段 {chunk.chunk_id + 1}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一名质性研究者。所有编码都必须有提供的访谈文本作为依据。"},
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
        Each theme should briefly synthesize related codes.

        Args:
            codes: List of Code objects

        Returns:
            List of Theme objects
        """
        codes_text = "\n".join([f"- {code.description}" for code in codes])

        prompt = f"""你是一名质性研究者，正在进行归纳式主题分析。

请将以下编码归纳为连贯的主题。

要求：
- 将相关编码归入更宽泛的主题
- 每个主题有清楚、具体的名称，描述用一句简短的话表达
- 主题应概括材料中的重要规律，并与其他主题有所区分
- 每个主题的 "codes" 字段列出其包含的原始编码描述，不改写这些描述
- 仅依据提供的编码归纳主题，不编造其中未出现的人群、场景或研究主题
- 主题名称和描述使用与编码相同的语言

编码：
{codes_text}

只返回符合以下结构的 JSON 对象，不添加解释或 Markdown：
{{
  "themes": [
    {{
      "name": "主题名称",
      "description": "用一句简短的话描述主题",
      "codes": ["属于该主题的原始编码描述"]
    }}
  ]
}}
"""

        if self.before_model_call:
            self.before_model_call("归纳主题")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一名质性研究者。所有主题都必须有提供的编码作为依据。"},
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
