import os
import re
from typing import Any

from generation.groq_client import create_chat_completion
from vectorstore.chroma_manager import ChromaManager
from retrieval.dense_search import dense_search


def generate_query_expansions(
    question: str,
    num_expansions: int = 3,
    model_name: str | None = None,
    api_key: str | None = None,
) -> list[str]:
    """
    Generate alternative search queries using a Groq-hosted LLM.

    Returns alternative queries only, not the original question.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if num_expansions <= 0:
        raise ValueError("num_expansions must be positive.")

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set in your environment."
        )

    model_name = model_name or os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    response = create_chat_completion(
        api_key,
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search-query expansion assistant. "
                    "Generate alternative search queries that preserve "
                    "the user's original intent while using different "
                    "wording, synonyms, and relevant terminology. "
                    "Preserve exact product names, error codes, IDs, "
                    "and technical identifiers. Do not introduce "
                    "unsupported assumptions. "
                    "Return only the requested queries, one per line, "
                    "without explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {question.strip()}\n\n"
                    f"Generate {num_expansions} alternative search queries."
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=min(200, num_expansions * 50),
    )

    content = response.choices[0].message.content

    if not content:
        return []

    expansions = []
    seen = {question.strip().casefold()}

    for line in content.splitlines():
        # Remove bullets or numbering from the model's response.
        query = re.sub(
            r"^\s*(?:(?:[-*•])|(?:\d+[.)]))\s*",
            "",
            line,
        ).strip().strip('"').strip("'")

        if not query:
            continue

        key = query.casefold()

        if key not in seen:
            seen.add(key)
            expansions.append(query)

        if len(expansions) >= num_expansions:
            break

    return expansions


def fuse_query_rankings(
    ranked_results: list[tuple[str, list[dict[str, Any]]]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Combine result rankings from multiple queries using RRF.

    Each item in ranked_results is:
        (query_text, results_for_that_query)
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    fused: dict[str, dict[str, Any]] = {}

    for query, results in ranked_results:
        seen_in_query = set()

        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]

            # Avoid counting the same chunk twice in one ranking.
            if chunk_id in seen_in_query:
                continue

            seen_in_query.add(chunk_id)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "matched_queries": [],
                    "query_ranks": {},
                }

            fused[chunk_id]["rrf_score"] += 1.0 / (
                rrf_k + rank
            )

            fused[chunk_id]["matched_queries"].append(query)
            fused[chunk_id]["query_ranks"][query] = rank

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def query_expansion_search(
    question: str,
    chroma_manager: ChromaManager,
    chunking_strategy: str,
    embedding_model: str,
    tenant_id: str,
    top_k: int = 5,
    candidate_k: int = 10,
    num_expansions: int = 3,
    filters: dict[str, Any] | None = None,
    model_name: str | None = None,
    rrf_k: int = 60,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search using the original question and its expanded variants.

    Workflow:
        1. Generate alternative queries.
        2. Run dense search for every query.
        3. Combine result rankings using RRF.
        4. Return the top-k chunks.

    All searches use the same embedding model, collection,
    tenant restriction, and applicable metadata filters.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("A valid tenant_id is required.")

    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("top_k and candidate_k must be positive.")

    original_question = question.strip()

    # Generate alternative search queries.
    expansions = generate_query_expansions(
        question=original_question,
        num_expansions=num_expansions,
        model_name=model_name,
        api_key=api_key,
    )

    # Always search with the original question too.
    queries = list(dict.fromkeys(
        [original_question] + expansions
    ))

    # Run dense retrieval for each query.
    ranked_results = []

    for query in queries:
        results = dense_search(
            question=query,
            chroma_manager=chroma_manager,
            chunking_strategy=chunking_strategy,
            embedding_model=embedding_model,
            tenant_id=tenant_id,
            top_k=candidate_k,
            filters=filters,
        )

        ranked_results.append((query, results))

    # Fuse the rankings into one result list.
    combined_results = fuse_query_rankings(
        ranked_results=ranked_results,
        rrf_k=rrf_k,
    )

    return combined_results[:top_k]