from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document


def _normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def get_evidence_passages(test_case: Mapping[str, Any]) -> list[str]:
    """Return user-supplied source evidence quotes without requiring chunk IDs."""
    evidence = test_case.get("evidence")
    passages = test_case.get("evidence_passages")

    if evidence is not None and passages is not None:
        raise ValueError(
            "Use either 'evidence' or 'evidence_passages', not both."
        )

    raw_passages = passages if passages is not None else evidence
    if raw_passages is None:
        return []
    if isinstance(raw_passages, str):
        raw_passages = [raw_passages]
    if not isinstance(raw_passages, Sequence):
        raise TypeError("'evidence' must be a string or a list of strings.")

    normalized_passages = []
    for passage in raw_passages:
        if not isinstance(passage, str) or not passage.strip():
            raise ValueError("Evidence passages must be non-empty strings.")
        normalized_passages.append(passage.strip())
    return normalized_passages


def map_evidence_to_chunks(
    evidence_passages: Sequence[str],
    chunks: Sequence[Document],
) -> dict[str, Any]:
    """Map source quotes to generated chunk IDs for one chunking strategy."""
    if not evidence_passages:
        return {
            "relevant_chunk_ids": [],
            "status": "unavailable",
            "reason": "No evidence quote was supplied in the evaluation case.",
            "matched_evidence_count": 0,
            "unmatched_evidence": [],
        }

    normalized_quotes = [
        _normalize_evidence(passage)
        for passage in evidence_passages
    ]
    relevant_ids: set[str] = set()
    matched_quotes: set[str] = set()

    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Every searchable chunk must have a generated chunk_id.")
        normalized_chunk = _normalize_evidence(chunk.page_content)
        for quote in normalized_quotes:
            if quote in normalized_chunk:
                relevant_ids.add(chunk_id)
                matched_quotes.add(quote)

    unmatched = [
        passage
        for passage, normalized in zip(evidence_passages, normalized_quotes)
        if normalized not in matched_quotes
    ]
    if unmatched:
        return {
            "relevant_chunk_ids": sorted(relevant_ids),
            "status": "unavailable",
            "reason": (
                "One or more evidence quotes did not match a generated chunk; "
                "metrics are withheld to avoid incomplete ground truth."
            ),
            "matched_evidence_count": len(matched_quotes),
            "unmatched_evidence": unmatched,
        }

    return {
        "relevant_chunk_ids": sorted(relevant_ids),
        "status": "available",
        "reason": None,
        "matched_evidence_count": len(matched_quotes),
        "unmatched_evidence": [],
    }


def resolve_relevant_chunks(
    test_case: Mapping[str, Any],
    strategy: str,
    config_id: str,
    chunks: Sequence[Document],
) -> dict[str, Any]:
    """Resolve relevance from evidence quotes, retaining legacy ID labels."""
    passages = get_evidence_passages(test_case)
    if passages:
        return map_evidence_to_chunks(passages, chunks)

    by_strategy = test_case.get("relevant_chunk_ids_by_strategy", {})
    if isinstance(by_strategy, Mapping) and strategy in by_strategy:
        ids = list(by_strategy[strategy] or [])
    else:
        by_config = test_case.get("relevant_chunk_ids_by_config", {})
        if isinstance(by_config, Mapping) and config_id in by_config:
            ids = list(by_config[config_id] or [])
        else:
            ids = list(test_case.get("relevant_chunk_ids", []) or [])

    if not ids:
        return {
            "relevant_chunk_ids": [],
            "status": "unavailable",
            "reason": "No evidence quote or relevant chunk labels were supplied.",
            "matched_evidence_count": 0,
            "unmatched_evidence": [],
        }

    available_ids = {
        str(chunk.metadata.get("chunk_id"))
        for chunk in chunks
        if chunk.metadata.get("chunk_id")
    }
    invalid_ids = sorted({str(chunk_id) for chunk_id in ids} - available_ids)
    if invalid_ids:
        return {
            "relevant_chunk_ids": [],
            "status": "unavailable",
            "reason": "One or more supplied chunk labels do not exist in this run.",
            "matched_evidence_count": 0,
            "unmatched_evidence": invalid_ids,
        }

    return {
        "relevant_chunk_ids": sorted({str(chunk_id) for chunk_id in ids}),
        "status": "available",
        "reason": None,
        "matched_evidence_count": 0,
        "unmatched_evidence": [],
    }
