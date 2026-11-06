from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


@lru_cache(maxsize=2)
def load_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> CrossEncoder:
    """
    Load the cross-encoder once and reuse it across queries.
    """

    return CrossEncoder(model_name)


def rerank_results(
    question: str,
    candidate_results: Sequence[dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """
    Rerank retrieved document chunks using a cross-encoder.

    Args:
        question:
            Original user question.

        candidate_results:
            Results from dense_search.py, hybrid_search.py,
            hyde.py, or query_expansion.py.

        top_k:
            Number of final chunks to return.

        model_name:
            Cross-encoder model used for relevance scoring.

        batch_size:
            Number of question-chunk pairs processed per batch.

    Returns:
        The top_k candidate chunks, sorted by reranker score.
        Each result retains its original information and includes
        'reranker_score' and 'rerank_rank'.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    candidates = list(candidate_results)

    if not candidates:
        return []

    # Build (question, candidate_chunk) pairs.
    pairs = []
    valid_candidates = []

    for candidate in candidates:
        if "text" in candidate:
            chunk_text = candidate["text"]
        elif "page_content" in candidate:
            chunk_text = candidate["page_content"]
        elif "document" in candidate:
            chunk_text = candidate["document"]
        else:
            raise ValueError(
                "Each candidate must contain 'text', "
                "'page_content', or 'document'."
            )

        if not isinstance(chunk_text, str) or not chunk_text.strip():
            continue

        pairs.append((question.strip(), chunk_text))
        valid_candidates.append(candidate)

    if not valid_candidates:
        return []

    # Load the cross-encoder and score every candidate.
    model = load_reranker(model_name)

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    scores = np.asarray(scores).reshape(-1)

    if len(scores) != len(valid_candidates):
        raise RuntimeError(
            "The number of relevance scores does not "
            "match the number of candidates."
        )

    # Preserve all original fields and attach the new score.
    reranked_results = []

    for candidate, score in zip(valid_candidates, scores):
        result = dict(candidate)
        result["reranker_score"] = float(score)
        reranked_results.append(result)

    # Higher cross-encoder scores indicate greater relevance.
    reranked_results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True,
    )

    # Return only the final top_k chunks.
    final_results = reranked_results[:top_k]

    for rank, result in enumerate(final_results, start=1):
        result["rerank_rank"] = rank

    return final_results