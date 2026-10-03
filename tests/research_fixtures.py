"""Shared synthetic evidence and model responses for research-record tests."""

import json
from types import SimpleNamespace

from evidence_fixtures import matched_code
from research_records import SourceMetadata


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])


def code(code_id=0, excerpt="在内部保留资料", source_id="S1", **kwargs):
    return matched_code(code_id=code_id, name="控制信息", description="控制信息", source_chunks=[0],
                excerpt=excerpt, source_start=0, source_end=len(excerpt),
                source=SourceMetadata(source_id=source_id), **kwargs)
