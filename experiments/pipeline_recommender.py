from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


# Metrics are weighted within their own evaluation group. The weights are
# defaults for this project, not universal definitions of RAG quality.
DEFAULT_RETRIEVAL_WEIGHTS = {
    "recall": 0.30,
    "ndcg": 0.25,
    "mrr": 0.20,
    "hit_rate": 0.15,
    "precision": 0.10,
}

DEFAULT_RESPONSE_WEIGHTS = {
    "faithfulness": 0.25,
    "answer_relevance": 0.20,
    "correctness": 0.25,
    "completeness": 0.15,
    "unsupported_claim_rate": 0.10,  # inverted when scoring
    "citation_accuracy": 0.05,
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _valid_score(value: Any) -> float | None:
    """Return a finite score in [0, 1], or None if unavailable/invalid."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    return score


def _largest_k_metric(
    metrics: Mapping[str, Any],
    prefix: str,
) -> tuple[str, float] | None:
    """Find prefix@K, preferring the largest available K."""
    candidates: list[tuple[int, str, float]] = []

    for key, value in metrics.items():
        if not key.startswith(prefix + "@"):
            continue
        suffix = key.split("@", 1)[1]
        if not suffix.isdigit():
            continue
        score = _valid_score(value)
        if score is not None:
            candidates.append((int(suffix), key, score))

    if not candidates:
        return None

    _, key, score = max(candidates, key=lambda item: item[0])
    return key, score


def _select_retrieval_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, tuple[str, float]]:
    """Select one usable score per retrieval-metric family."""
    selected: dict[str, tuple[str, float]] = {}

    # Prefer full-ranking MRR if supplied; otherwise use the largest MRR@K.
    mrr = _valid_score(metrics.get("mrr"))
    if mrr is not None:
        selected["mrr"] = ("mrr", mrr)
    else:
        chosen_mrr = _largest_k_metric(metrics, "mrr")
        if chosen_mrr is not None:
            selected["mrr"] = chosen_mrr

    for family in ("recall", "ndcg", "hit_rate", "precision"):
        chosen = _largest_k_metric(metrics, family)
        if chosen is not None:
            selected[family] = chosen

    return selected


def _weighted_mean(
    available: Mapping[str, tuple[float, float]],
) -> float | None:
    """Weighted average, renormalized over available metrics only."""
    if not available:
        return None

    denominator = sum(weight for _, weight in available.values())
    if denominator <= 0:
        return None

    return sum(score * weight for score, weight in available.values()) / denominator


def _score_retrieval(
    metrics: Mapping[str, Any] | None,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    metrics = metrics or {}
    selected = _select_retrieval_metrics(metrics)

    available = {
        family: (score, weights.get(family, 0.0))
        for family, (_, score) in selected.items()
        if weights.get(family, 0.0) > 0
    }

    detail = {
        family: {"metric": metric_name, "score": score}
        for family, (metric_name, score) in selected.items()
    }

    evidence_score = _weighted_mean(available)
    if evidence_score is not None:
        return {
            "score": evidence_score,
            "metrics_used": detail,
            "source": "evidence_mapped_chunk_metrics",
        }

    return {
        "score": None,
        "metrics_used": {},
        "source": None,
    }


def _score_response(
    metrics: Mapping[str, Any] | None,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    metrics = metrics or {}
    available: dict[str, tuple[float, float]] = {}
    detail: dict[str, Any] = {}

    for metric_name, weight in weights.items():
        score = _valid_score(metrics.get(metric_name))
        if score is None or weight <= 0:
            continue

        # Lower unsupported-claim rate is better, so invert it into quality.
        quality_score = (
            1.0 - score
            if metric_name == "unsupported_claim_rate"
            else score
        )

        available[metric_name] = (quality_score, weight)
        detail[metric_name] = {
            "raw_score": score,
            "quality_score": quality_score,
        }

    return {
        "score": _weighted_mean(available),
        "metrics_used": detail,
    }


def _pipeline_label(pipeline: Mapping[str, Any]) -> str:
    parts = [
        str(pipeline.get("chunking_strategy", "unknown_chunking")),
        str(pipeline.get("embedding_model", "unknown_embedding")),
        str(pipeline.get("retrieval_method", "unknown_retrieval")),
    ]
    if pipeline.get("reranker_enabled"):
        parts.append("reranked")
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def recommend_pipeline(
    report: Mapping[str, Any] | str | Path,
    *,
    retrieval_weight: float = 0.60,
    response_weight: float = 0.40,
    retrieval_metric_weights: Mapping[str, float] | None = None,
    response_metric_weights: Mapping[str, float] | None = None,
    require_response_metrics: bool = False,
) -> dict[str, Any]:
    """Rank evaluated RAG pipelines and recommend the strongest observed one.

    Args:
        report:
            The dictionary returned by `run_experiments`, or a path to its
            JSON report.
        retrieval_weight / response_weight:
            Relative importance of retrieval and response quality. When a
            pipeline lacks response metrics, available group weights are
            renormalized; response weight is not treated as zero quality.
        retrieval_metric_weights:
            Weights for Recall, nDCG, MRR, Hit Rate, and Precision.
        response_metric_weights:
            Weights for faithfulness, relevance, correctness, completeness,
            unsupported-claim rate (inverted), and citation accuracy.
        require_response_metrics:
            If True, pipelines without a response-quality score are excluded.

    Returns:
        A dictionary with the recommendation, ranked pipelines, scoring
        details, and caveats. Scores are normalized to [0, 1] where available.
    """
    if not 0 <= retrieval_weight <= 1 or not 0 <= response_weight <= 1:
        raise ValueError("Group weights must be between 0 and 1.")
    if retrieval_weight + response_weight <= 0:
        raise ValueError("At least one group weight must be positive.")

    if isinstance(report, (str, Path)):
        report_path = Path(report)
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
    elif isinstance(report, Mapping):
        report_data = dict(report)
    else:
        raise TypeError("report must be a report dictionary or JSON file path.")

    pipelines = report_data.get("pipelines")
    if not isinstance(pipelines, list) or not pipelines:
        raise ValueError("The report must contain a non-empty 'pipelines' list.")

    ret_weights = dict(
        DEFAULT_RETRIEVAL_WEIGHTS
        if retrieval_metric_weights is None
        else retrieval_metric_weights
    )
    resp_weights = dict(
        DEFAULT_RESPONSE_WEIGHTS
        if response_metric_weights is None
        else response_metric_weights
    )

    for name, weight in {**ret_weights, **resp_weights}.items():
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"Weight for '{name}' must be non-negative.")

    ranked: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    for pipeline in pipelines:
        if not isinstance(pipeline, Mapping):
            continue

        retrieval = _score_retrieval(
            pipeline.get("mean_retrieval_metrics"),
            ret_weights,
        )
        response = _score_response(
            pipeline.get("mean_response_metrics"),
            resp_weights,
        )

        retrieval_score = retrieval["score"]
        response_score = response["score"]

        if require_response_metrics and response_score is None:
            excluded.append({
                "config_id": str(pipeline.get("config_id", "unknown")),
                "reason": "No valid response-quality metrics are available.",
            })
            continue

        available_groups: dict[str, tuple[float, float]] = {}
        if retrieval_score is not None and retrieval_weight > 0:
            available_groups["retrieval"] = (retrieval_score, retrieval_weight)
        if response_score is not None and response_weight > 0:
            available_groups["response"] = (response_score, response_weight)

        overall_score = _weighted_mean(available_groups)
        if overall_score is None:
            excluded.append({
                "config_id": str(pipeline.get("config_id", "unknown")),
                "reason": "No usable labelled retrieval or response metrics.",
            })
            continue

        ranked.append({
            "config_id": pipeline.get("config_id"),
            "label": _pipeline_label(pipeline),
            "collection_name": pipeline.get("collection_name"),
            "chunking_strategy": pipeline.get("chunking_strategy"),
            "embedding_model": pipeline.get("embedding_model"),
            "retrieval_method": pipeline.get("retrieval_method"),
            "reranker_enabled": pipeline.get("reranker_enabled", False),
            "number_of_searchable_chunks": pipeline.get(
                "number_of_searchable_chunks"
            ),
            "retrieval_score": retrieval_score,
            "retrieval_score_source": retrieval["source"],
            "response_score": response_score,
            "comparison_complete": (
                retrieval_score is not None and response_score is not None
            ),
            "overall_score": overall_score,
            "retrieval_metrics_used": retrieval["metrics_used"],
            "response_metrics_used": response["metrics_used"],
            "mean_retrieval_metrics": pipeline.get("mean_retrieval_metrics") or {},
            "mean_response_metrics": pipeline.get("mean_response_metrics"),
            "mean_retrieval_latency_seconds": pipeline.get(
                "mean_retrieval_latency_seconds"
            ),
            "mean_end_to_end_latency_seconds": pipeline.get(
                "mean_end_to_end_latency_seconds"
            ),
        })

    if not ranked:
        raise ValueError(
            "No pipelines could be scored. Check that the experiment report "
            "contains valid evidence-mapped retrieval metrics or response metrics."
        )

    ranked.sort(key=lambda item: item["overall_score"], reverse=True)

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    winner = ranked[0]
    notes = [
        "The recommendation is based only on the metrics present in the supplied experiment report.",
        "Metric weights are configurable design choices, not universal RAG standards.",
        "Retrieval metrics are calculated mathematically from retrieved IDs and relevance IDs mapped internally from source evidence quotes.",
        "Retrieval metrics are unavailable when evidence is missing or could not be matched to chunks; no relevance labels are fabricated.",
        "Latency is reported for comparison but is not included in the quality score. API cost is not currently estimated.",
    ]

    if winner["response_score"] is None:
        notes.append(
            "Response evaluation was unavailable; the ranking uses retrieval quality only and is not a full pipeline comparison."
        )
    if winner["retrieval_score"] is None:
        notes.append(
            "Retrieval quality was unavailable; the ranking uses response metrics only and is not a full pipeline comparison."
        )
    if any(not item["comparison_complete"] for item in ranked):
        notes.append(
            "Some pipeline scores omit a quality group; review comparison_complete before treating the ranking as a retrieval-and-response comparison."
        )
    if len(ranked) == 1:
        notes.append("Only one pipeline had usable metrics, so no meaningful comparison was possible.")

    return {
        "recommended_pipeline": winner,
        "ranking": ranked,
        "excluded_pipelines": excluded,
        "scoring_settings": {
            "retrieval_group_weight": retrieval_weight,
            "response_group_weight": response_weight,
            "retrieval_metric_weights": ret_weights,
            "response_metric_weights": resp_weights,
            "require_response_metrics": require_response_metrics,
        },
        "notes": notes,
    }


def save_recommendation(
    recommendation: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a recommendation report as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
