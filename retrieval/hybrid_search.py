import re
from typing import Any

from rank_bm25 import BM25Okapi

from vectorstore.chroma_manager import ChromaManager
from retrieval.dense_search import dense_search
from retrieval.metadata_filters import build_metadata_where


def tokenize(text: str) -> list[str]:
    """
    Convert text into searchable terms.

    Preserves identifiers such as TX-104, product-25,
    and API/v2 as individual tokens.
    """
    return re.findall(
        r"[a-zA-Z0-9_]+(?:[-./][a-zA-Z0-9_]+)*",
        text.lower(),
    )


def build_where_filter(
    tenant_id: str,
    chunking_strategy: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build metadata filters for BM25.

    Uses the same tenant restriction as dense retrieval.
    Hierarchical search indexes only child chunks.
    """

    return build_metadata_where(
        tenant_id=tenant_id,
        chunking_strategy=chunking_strategy,
        filters=filters,
    )


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Combine dense and BM25 rankings using Reciprocal Rank Fusion.

    RRF score = sum(1 / (rrf_k + rank))

    Ranks start at 1. A chunk appearing high in both result
    lists receives a higher combined score.
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    combined: dict[str, dict[str, Any]] = {}

    # Add dense search results.
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["id"]

        combined[chunk_id] = {
            **result,
            "dense_rank": rank,
            "bm25_rank": None,
            "bm25_score": 0.0,
            "rrf_score": 1.0 / (rrf_k + rank),
        }

    # Add BM25 results and combine ranks for duplicate chunks.
    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result["id"]
        contribution = 1.0 / (rrf_k + rank)

        if chunk_id in combined:
            combined[chunk_id]["bm25_rank"] = rank
            combined[chunk_id]["bm25_score"] = result["bm25_score"]
            combined[chunk_id]["rrf_score"] += contribution

        else:
            combined[chunk_id] = {
                **result,
                "distance": None,
                "dense_rank": None,
                "bm25_rank": rank,
                "rrf_score": contribution,
            }

    return sorted(
        combined.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def hybrid_search(
    question: str,
    chroma_manager: ChromaManager,
    chunking_strategy: str,
    embedding_model: str,
    tenant_id: str,
    top_k: int = 5,
    dense_k: int = 20,
    sparse_k: int = 20,
    rrf_k: int = 60,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant chunks using dense search + BM25 keyword search.

    Steps:
        1. Retrieve candidates using vector similarity.
        2. Fetch authorized chunks for BM25 search.
        3. Rank those chunks using keyword relevance.
        4. Combine both rankings with RRF.
        5. Return the top_k combined results.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("A valid tenant_id is required.")

    if top_k <= 0 or dense_k <= 0 or sparse_k <= 0:
        raise ValueError("top_k, dense_k and sparse_k must be positive.")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    question = question.strip()

    # ---------------------------------------------------------
    # 1. Dense retrieval using your existing implementation.
    # ---------------------------------------------------------
    dense_results = dense_search(
        question=question,
        chroma_manager=chroma_manager,
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
        tenant_id=tenant_id,
        top_k=dense_k,
        filters=filters,
    )

    # ---------------------------------------------------------
    # 2. Select the same ChromaDB collection for BM25.
    # ---------------------------------------------------------
    collection = chroma_manager.get_collection(
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
    )

    where_filter = build_where_filter(
        tenant_id=tenant_id,
        chunking_strategy=chunking_strategy,
        filters=filters,
    )

    stored = collection.get(
        where=where_filter,
        include=["documents", "metadatas"],
    )

    ids = stored["ids"]
    texts = stored["documents"] or []
    metadatas = stored["metadatas"] or []

    # Only use records containing actual text.
    corpus = []

    for chunk_id, text, metadata in zip(ids, texts, metadatas):
        if not text or not text.strip():
            continue

        tokens = tokenize(text)

        if not tokens:
            continue

        corpus.append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata or {},
            "tokens": tokens,
        })

    # If there are no searchable chunks, return dense results.
    if not corpus:
        return dense_results[:top_k]

    # ---------------------------------------------------------
    # 3. Build a BM25 keyword index and rank the chunks.
    # ---------------------------------------------------------
    tokenized_corpus = [
        item["tokens"] for item in corpus
    ]

    query_tokens = tokenize(question)

    if not query_tokens:
        return dense_results[:top_k]

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query_tokens)

    # A chunk must have a positive BM25 score to enter the
    # sparse ranking. This avoids adding irrelevant zero-score
    # chunks when the query has no matching terms.
    ranked_indices = sorted(
        range(len(corpus)),
        key=lambda index: float(bm25_scores[index]),
        reverse=True,
    )

    sparse_results = []

    for index in ranked_indices:
        score = float(bm25_scores[index])

        if score <= 0:
            continue

        item = corpus[index]

        sparse_results.append({
            "id": item["id"],
            "text": item["text"],
            "metadata": item["metadata"],
            "bm25_score": score,
        })

        if len(sparse_results) >= sparse_k:
            break

    # ---------------------------------------------------------
    # 4. Fuse the rankings and select the final top_k chunks.
    # ---------------------------------------------------------
    combined_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        rrf_k=rrf_k,
    )

    return combined_results[:top_k]