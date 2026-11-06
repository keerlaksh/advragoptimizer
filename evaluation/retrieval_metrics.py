from collections.abc import Collection, Mapping, Sequence
from math import log2
from typing import Any


DEFAULT_K_VALUES = (1, 3, 5)


def _get_ordered_chunk_ids(
    retrieved_results: Sequence[Any],
) -> list[str]:
    """Extract ranked chunk IDs from retrieval results."""

    chunk_ids = []
    seen = set()

    for result in retrieved_results:
        if isinstance(result, Mapping):
            chunk_id = result.get("id", result.get("chunk_id"))
        else:
            chunk_id = result

        if chunk_id is None:
            raise ValueError(
                "Each retrieval result must contain a chunk ID."
            )

        chunk_id = str(chunk_id)

        # Count a chunk only at its first occurrence.
        if chunk_id not in seen:
            chunk_ids.append(chunk_id)
            seen.add(chunk_id)

    return chunk_ids


def _validate_relevant_ids(
    relevant_chunk_ids: Collection[str],
) -> set[str]:
    """Validate and normalize the known relevant chunk IDs."""

    if isinstance(relevant_chunk_ids, str):
        raise TypeError(
            "Pass relevant chunk IDs as a collection, "
            "such as {'chunk_1', 'chunk_2'}, not a single string."
        )

    relevant = {
        str(chunk_id)
        for chunk_id in relevant_chunk_ids
    }

    if any(not chunk_id.strip() for chunk_id in relevant):
        raise ValueError("Relevant chunk IDs cannot be empty strings.")

    return relevant


def hit_rate_at_k(
    retrieved_results: Sequence[Any],
    relevant_chunk_ids: Collection[str],
    k: int = 5,
) -> float | None:
    """
    Return 1 if at least one relevant chunk appears in the top k;
    otherwise return 0.

    Returns None when no relevant chunks have been labelled.
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    relevant = _validate_relevant_ids(relevant_chunk_ids)

    if not relevant:
        return None

    retrieved_ids = _get_ordered_chunk_ids(retrieved_results)
    top_ids = retrieved_ids[:k]

    return float(any(chunk_id in relevant for chunk_id in top_ids))


def precision_at_k(
    retrieved_results: Sequence[Any],
    relevant_chunk_ids: Collection[str],
    k: int = 5,
) -> float | None:
    """
    Calculate the fraction of the top k positions that are relevant.

    Missing result positions count as non-relevant, so the
    denominator remains k.

    Returns None when no relevant chunks have been labelled.
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    relevant = _validate_relevant_ids(relevant_chunk_ids)

    if not relevant:
        return None

    retrieved_ids = _get_ordered_chunk_ids(retrieved_results)
    top_ids = retrieved_ids[:k]

    relevant_count = sum(
        chunk_id in relevant for chunk_id in top_ids
    )

    return relevant_count / k


def recall_at_k(
    retrieved_results: Sequence[Any],
    relevant_chunk_ids: Collection[str],
    k: int = 5,
) -> float | None:
    """
    Calculate the fraction of all labelled relevant chunks retrieved
    within the top k results.

    Returns None when no relevant chunks have been labelled.
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    relevant = _validate_relevant_ids(relevant_chunk_ids)

    if not relevant:
        return None

    retrieved_ids = _get_ordered_chunk_ids(retrieved_results)
    top_ids = retrieved_ids[:k]

    relevant_retrieved = sum(
        chunk_id in relevant for chunk_id in top_ids
    )

    return relevant_retrieved / len(relevant)


def reciprocal_rank_at_k(
    retrieved_results: Sequence[Any],
    relevant_chunk_ids: Collection[str],
    k: int = 5,
) -> float | None:
    """
    Return the reciprocal rank of the first relevant chunk within
    the top k results.

    Example:
        First relevant chunk at rank 1 -> 1.0
        First relevant chunk at rank 2 -> 0.5
        No relevant chunk in the top k -> 0.0

    Returns None when no relevant chunks have been labelled.
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    relevant = _validate_relevant_ids(relevant_chunk_ids)

    if not relevant:
        return None

    retrieved_ids = _get_ordered_chunk_ids(retrieved_results)

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    retrieved_results: Sequence[Any],
    relevant_chunk_ids: Collection[str],
) -> float | None:
    """
    Calculate the reciprocal rank of the first relevant result
    over the complete returned ranking for one question.

    Dataset-level MRR is the mean of these per-question scores.
    """

    if not retrieved_results:
        relevant = _validate_relevant_ids(relevant_chunk_ids)
        return None if not relevant else 0.0

    # Use the full returned ranking as the cutoff.
    return reciprocal_rank_at_k(
        retrieved_results,
        relevant_chunk_ids,
        k=len(retrieved_results),
    )


def ndcg_at_k(
    retrieved_results: Sequence[Any],
    relevant_chunk_ids: Collection[str],
    k: int = 5,
) -> float | None:
    """
    Calculate binary-relevance Normalized Discounted Cumulative Gain.

    Relevant chunks receive gain 1; non-relevant chunks receive gain 0.
    Relevant chunks ranked higher contribute more to DCG.

    Returns None when no relevant chunks have been labelled.
    """

    if k <= 0:
        raise ValueError("k must be positive.")

    relevant = _validate_relevant_ids(relevant_chunk_ids)

    if not relevant:
        return None

    retrieved_ids = _get_ordered_chunk_ids(retrieved_results)
    top_ids = retrieved_ids[:k]

    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, chunk_id in enumerate(top_ids, start=1)
        if chunk_id in relevant
    )

    # Ideal ranking puts all known relevant chunks first.
    ideal_relevant_count = min(k, len(relevant))

    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    retrieved_results: Sequence[Any],
    relevant_chunk_ids: Collection[str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float | None]:
    """
    Calculate retrieval metrics for one question and one configuration.

    `retrieved_results` must be in ranked order, best result first.
    It can contain chunk ID strings or dictionaries returned by
    your search modules.
    """

    relevant = _validate_relevant_ids(relevant_chunk_ids)

    if not k_values:
        raise ValueError("Provide at least one k value.")

    if any(k <= 0 for k in k_values):
        raise ValueError("All k values must be positive.")

    if not relevant:
        return {
            metric_name: None
            for k in k_values
            for metric_name in (
                f"hit_rate@{k}",
                f"precision@{k}",
                f"recall@{k}",
                f"mrr@{k}",
                f"ndcg@{k}",
            )
        } | {"mrr": None}

    metrics: dict[str, float | None] = {}

    for k in k_values:
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(
            retrieved_results, relevant, k
        )
        metrics[f"precision@{k}"] = precision_at_k(
            retrieved_results, relevant, k
        )
        metrics[f"recall@{k}"] = recall_at_k(
            retrieved_results, relevant, k
        )
        metrics[f"mrr@{k}"] = reciprocal_rank_at_k(
            retrieved_results, relevant, k
        )
        metrics[f"ndcg@{k}"] = ndcg_at_k(
            retrieved_results, relevant, k
        )

    metrics["mrr"] = mean_reciprocal_rank(
        retrieved_results, relevant
    )

    return metrics


def evaluate_retrieval_dataset(
    test_cases: Sequence[Mapping[str, Any]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """
    Aggregate retrieval results across multiple test questions.

    Each test case must contain:
        - retrieved_results: ranked results from your retriever
        - relevant_chunk_ids: independently verified relevant chunk IDs

    Optional:
        - question: the question text, included in per-question output

    Test cases without relevant_chunk_ids are reported but excluded
    from the metric averages. They should be handled separately as
    unanswerable or not-yet-labelled cases.
    """

    per_question = []

    for index, test_case in enumerate(test_cases, start=1):
        if "retrieved_results" not in test_case:
            raise ValueError(
                f"Test case {index} is missing 'retrieved_results'."
            )

        if "relevant_chunk_ids" not in test_case:
            raise ValueError(
                f"Test case {index} is missing 'relevant_chunk_ids'."
            )

        metrics = evaluate_retrieval(
            retrieved_results=test_case["retrieved_results"],
            relevant_chunk_ids=test_case["relevant_chunk_ids"],
            k_values=k_values,
        )

        per_question.append({
            "question": test_case.get("question", f"Question {index}"),
            "metrics": metrics,
        })

    metric_names = (
        list(per_question[0]["metrics"].keys())
        if per_question
        else [
            metric_name
            for k in k_values
            for metric_name in (
                f"hit_rate@{k}",
                f"precision@{k}",
                f"recall@{k}",
                f"mrr@{k}",
                f"ndcg@{k}",
            )
        ] + ["mrr"]
    )

    mean_metrics: dict[str, float | None] = {}

    for metric_name in metric_names:
        values = [
            item["metrics"][metric_name]
            for item in per_question
            if item["metrics"].get(metric_name) is not None
        ]

        mean_metrics[metric_name] = (
            sum(values) / len(values)
            if values
            else None
        )

    evaluated_count = sum(
        any(value is not None for value in item["metrics"].values())
        for item in per_question
    )

    return {
        "total_questions": len(test_cases),
        "evaluated_questions": evaluated_count,
        "per_question": per_question,
        "mean_metrics": mean_metrics,
    }