"""Synthetic evidence that has actually been checked against a source body."""

from agents.generation_agent import Code
from evidence import source_document, validate_evidence
from research_records import SourceMetadata


def matched_code(**values):
    excerpt = values.get("excerpt") or values.get("description") or values.get("name") or "合成原话"
    start = values.get("source_start", 0)
    code = Code(**{"source_chunks": [], "source": SourceMetadata(source_id="fixture"),
                   "excerpt": excerpt, "source_start": start, "source_end": start + len(excerpt), **values})
    body = "_" * start + excerpt
    validate_evidence([code], [source_document(code.source.source_id, body)])
    code.merged_into = values.get("merged_into")
    code.merged_from = list(values.get("merged_from") or [])
    return code
