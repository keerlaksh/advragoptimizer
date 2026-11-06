from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from langchain_core.documents import Document

from chunking.fixed_size import chunk_documents as fixed_size_chunk
from chunking.semantic import chunk_documents as semantic_chunk
from chunking.hierarchical import (
    chunk_documents as hierarchical_chunk,
)
from chunking.structure_aware import (
    chunk_documents as structure_aware_chunk,
)
from vectorstore.chroma_manager import ChromaManager
from retrieval.dense_search import dense_search
from retrieval.hybrid_search import hybrid_search
from retrieval.hyde import hyde_search
from retrieval.query_expansion import query_expansion_search
from retrieval.reranker import rerank_results
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.evidence_mapping import resolve_relevant_chunks
from generation.answer_generator import generate_answer
from evaluation.response_metrics import evaluate_response


CHUNKING_STRATEGIES = (
    "fixed_size",
    "semantic",
    "hierarchical",
    "structure_aware",
)

EMBEDDING_MODELS = (
    "minilm",
    "e5-small",
    "bge-base",
)

RETRIEVAL_METHODS = (
    "dense",
    "hybrid",
    "hyde",
    "query_expansion",
)

DEFAULT_CHUNKING_OPTIONS: dict[str, dict[str, Any]] = {
    "fixed_size": {
        "chunk_size": 1000,
        "chunk_overlap": 150,
    },
    "semantic": {
        "breakpoint_percentile": 80.0,
        "max_chunk_size": 1200,
    },
    "hierarchical": {
        "parent_chunk_size": 2000,
        "parent_overlap": 100,
        "child_chunk_size": 500,
        "child_overlap": 75,
    },
    "structure_aware": {
        "chunk_size": 1200,
        "chunk_overlap": 100,
    },
}


def _chunk_for_strategy(
    documents: Sequence[Document],
    strategy: str,
    options: Mapping[str, Any] | None = None,
) -> tuple[list[Document], list[Document]]:
    """Return (searchable chunks, parent chunks)."""
    options = dict(options or {})

    if strategy == "fixed_size":
        return fixed_size_chunk(documents, **options), []

    if strategy == "semantic":
        return semantic_chunk(documents, **options), []

    if strategy == "hierarchical":
        result = hierarchical_chunk(documents, **options)
        return result.child_chunks, result.parent_chunks

    if strategy == "structure_aware":
        return structure_aware_chunk(documents, **options), []

    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def _retrieve(
    *,
    method: str,
    question: str,
    chroma_manager: ChromaManager,
    chunking_strategy: str,
    embedding_model: str,
    tenant_id: str,
    top_k: int,
    candidate_k: int,
    filters: dict[str, Any] | None,
    use_reranker: bool,
    reranker_model: str | None,
    groq_api_key: str | None,
) -> list[dict[str, Any]]:
    """Run one retrieval method and optionally rerank its candidates."""
    fetch_k = candidate_k if use_reranker else top_k

    common = {
        "question": question,
        "chroma_manager": chroma_manager,
        "chunking_strategy": chunking_strategy,
        "embedding_model": embedding_model,
        "tenant_id": tenant_id,
        "filters": filters,
    }

    if method == "dense":
        results = dense_search(
            **common,
            top_k=fetch_k,
        )
    elif method == "hybrid":
        results = hybrid_search(
            **common,
            top_k=fetch_k,
            dense_k=fetch_k,
            sparse_k=fetch_k,
        )
    elif method == "hyde":
        results = hyde_search(
            **common,
            top_k=fetch_k,
            api_key=groq_api_key,
        )
    elif method == "query_expansion":
        results = query_expansion_search(
            **common,
            top_k=fetch_k,
            candidate_k=fetch_k,
            api_key=groq_api_key,
        )
    else:
        raise ValueError(
            f"Unsupported retrieval method '{method}'. "
            f"Choose from {RETRIEVAL_METHODS}."
        )

    if use_reranker and results:
        return rerank_results(
            question=question,
            candidate_results=results,
            top_k=top_k,
            **({"model_name": reranker_model} if reranker_model else {}),
        )

    return results[:top_k]


def _mean_metrics(
    question_results: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> dict[str, float | None]:
    """Average numeric metrics while skipping unavailable (None) values."""
    metric_names: set[str] = set()

    for row in question_results:
        metrics = row.get(metric_key) or {}
        metric_names.update(metrics.keys())

    averages: dict[str, float | None] = {}
    for name in sorted(metric_names):
        values = [
            float((row.get(metric_key) or {})[name])
            for row in question_results
            if (row.get(metric_key) or {}).get(name) is not None
            and isinstance((row.get(metric_key) or {}).get(name), (int, float))
            and not isinstance((row.get(metric_key) or {}).get(name), bool)
        ]
        averages[name] = mean(values) if values else None

    return averages


def _retrieval_evaluation_summary(
    question_results: Sequence[Mapping[str, Any]],
) -> dict[str, int | str]:
    """Summarize per-question label coverage without implying all-or-nothing."""
    total = len(question_results)
    available = sum(
        1
        for row in question_results
        if (row.get("retrieval_evaluation") or {}).get("status") == "available"
    )
    unavailable = total - available

    if total and available == total:
        status = "available"
    elif available:
        status = "partially_available"
    else:
        status = "unavailable"

    return {
        "retrieval_evaluation_status": status,
        "retrieval_evaluation_available_questions": available,
        "retrieval_evaluation_unavailable_questions": unavailable,
        "retrieval_evaluation_total_questions": total,
    }


def run_experiments(
    documents: Sequence[Document],
    test_cases: Sequence[Mapping[str, Any]],
    tenant_id: str,
    *,
    persist_directory: str = "chroma_data",
    chunking_strategies: Sequence[str] = CHUNKING_STRATEGIES,
    embedding_models: Sequence[str] = EMBEDDING_MODELS,
    retrieval_methods: Sequence[str] = ("dense", "hybrid"),
    top_k: int = 5,
    candidate_k: int = 20,
    use_reranker: bool = False,
    reranker_model: str | None = None,
    groq_api_key: str | None = None,
    chunking_options: Mapping[str, Mapping[str, Any]] | None = None,
    k_values: Sequence[int] = (1, 3, 5),
    max_questions: int | None = None,
    output_path: str | Path | None = None,
    progress_callback: Callable[[str, str, str], None] | None = None,
) -> dict[str, Any]:
    """Run configurable RAG experiments across chunking and embedding choices.

    `documents` should be the output of `load_document()` and each document
    must already have the authorized `tenant_id` in its metadata.

    Each test case should contain `question`. Add `evidence` as a source quote
    (or `evidence_passages` as a list of quotes) to map relevance automatically
    onto the generated chunks for each chunking strategy. Chunk IDs are internal.
    Without evidence or legacy relevant-ID labels, retrieval metrics remain
    unavailable rather than being fabricated.

    Answer generation and response evaluation always run and require Groq.
    Dense, hybrid, reranking, and retrieval metrics themselves are local.
    """
    if not documents:
        raise ValueError("No source documents were supplied.")
    if not test_cases:
        raise ValueError("No test cases were supplied.")
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("A valid tenant_id is required.")
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("Require top_k > 0 and candidate_k >= top_k.")
    if max_questions is not None and max_questions <= 0:
        raise ValueError("max_questions must be positive when provided.")
    groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key or not groq_api_key.strip():
        raise ValueError(
            "A Groq API key is required because answer generation and response "
            "evaluation run for every experiment."
        )

    for doc in documents:
        if doc.metadata.get("tenant_id") != tenant_id:
            raise ValueError(
                "Each document's tenant_id must match the experiment tenant_id. "
                "Load documents with the correct trusted tenant context."
            )

    strategies = tuple(chunking_strategies)
    models = tuple(embedding_models)
    methods = tuple(retrieval_methods)

    invalid_strategies = set(strategies) - set(CHUNKING_STRATEGIES)
    invalid_models = set(models) - set(EMBEDDING_MODELS)
    invalid_methods = set(methods) - set(RETRIEVAL_METHODS)
    if invalid_strategies:
        raise ValueError(f"Unknown chunking strategies: {sorted(invalid_strategies)}")
    if invalid_models:
        raise ValueError(f"Unknown embedding models: {sorted(invalid_models)}")
    if invalid_methods:
        raise ValueError(f"Unknown retrieval methods: {sorted(invalid_methods)}")
    if not strategies or not models or not methods:
        raise ValueError(
            "Select at least one chunking strategy, embedding model, and "
            "retrieval method."
        )

    def report_progress(stage: str, state: str, message: str) -> None:
        if progress_callback is not None:
            progress_callback(stage, state, message)

    cases = list(test_cases[:max_questions] if max_questions else test_cases)
    options_by_strategy = dict(DEFAULT_CHUNKING_OPTIONS)
    if chunking_options:
        for strategy, options in chunking_options.items():
            if strategy not in CHUNKING_STRATEGIES:
                raise ValueError(f"Unknown chunking option key: {strategy}")
            options_by_strategy[strategy] = {
                **options_by_strategy[strategy],
                **dict(options),
            }

    chroma_manager = ChromaManager(
        persist_directory=persist_directory,
    )

    # Chunk each strategy only once. The resulting chunks are reused across
    # embedding models so each strategy/model comparison uses identical text.
    chunks_by_strategy: dict[str, tuple[list[Document], list[Document]]] = {}
    report_progress("chunking", "running", "Creating chunks for selected strategies.")
    for strategy in strategies:
        report_progress("chunking", "running", f"Chunking with {strategy}.")
        searchable, parents = _chunk_for_strategy(
            documents,
            strategy,
            options_by_strategy[strategy],
        )
        if not searchable:
            raise ValueError(
                f"Chunking strategy '{strategy}' produced no searchable chunks."
            )
        chunks_by_strategy[strategy] = (searchable, parents)
    report_progress(
        "chunking",
        "complete",
        f"Created chunks for {len(strategies)} chunking strategy/strategies.",
    )

    pipeline_reports: list[dict[str, Any]] = []
    total_queries = len(strategies) * len(models) * len(methods) * len(cases)
    query_number = 0

    for strategy in strategies:
        searchable_chunks, parent_chunks = chunks_by_strategy[strategy]

        for embedding_model in models:
            report_progress(
                "embedding_storage",
                "running",
                f"Embedding {strategy} chunks with {embedding_model} and storing them in ChromaDB.",
            )
            collection_name = chroma_manager.store_chunks(
                chunks=searchable_chunks,
                parent_chunks=parent_chunks,
                chunking_strategy=strategy,
                embedding_model=embedding_model,
                replace_existing=True,
            )

            config_id = chroma_manager.get_collection_name(
                strategy,
                embedding_model,
            )
            report_progress(
                "embedding_storage",
                "complete",
                f"Stored {strategy} + {embedding_model} in ChromaDB.",
            )

            for retrieval_method in methods:
                question_reports: list[dict[str, Any]] = []

                for test_case in cases:
                    question = test_case.get("question")
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError(
                            "Every test case must contain a non-empty 'question'."
                        )

                    filters = test_case.get("filters")
                    if filters is not None and not isinstance(filters, dict):
                        raise TypeError("Test case 'filters' must be a dictionary.")

                    query_number += 1
                    query_status = (
                        f"{query_number}/{total_queries}: {strategy} + "
                        f"{embedding_model} + {retrieval_method}"
                    )
                    report_progress(
                        "retrieval",
                        "running",
                        f"Retrieving {query_status}.",
                    )
                    pipeline_started = perf_counter()
                    retrieval_started = perf_counter()
                    retrieved = _retrieve(
                        method=retrieval_method,
                        question=question,
                        chroma_manager=chroma_manager,
                        chunking_strategy=strategy,
                        embedding_model=embedding_model,
                        tenant_id=tenant_id,
                        top_k=top_k,
                        candidate_k=candidate_k,
                        filters=filters,
                        use_reranker=use_reranker,
                        reranker_model=reranker_model,
                        groq_api_key=groq_api_key,
                    )
                    retrieval_latency = perf_counter() - retrieval_started
                    report_progress(
                        "retrieval",
                        "complete",
                        f"Retrieved {len(retrieved)} chunk(s) for {query_status}.",
                    )

                    report_progress(
                        "retrieval_metrics",
                        "running",
                        f"Mapping evidence and calculating metrics for {query_status}.",
                    )
                    relevance = resolve_relevant_chunks(
                        test_case=test_case,
                        strategy=strategy,
                        config_id=config_id,
                        chunks=searchable_chunks,
                    )

                    retrieval_scores = evaluate_retrieval(
                        retrieved_results=retrieved,
                        relevant_chunk_ids=(
                            relevance["relevant_chunk_ids"]
                            if relevance["status"] == "available"
                            else []
                        ),
                        k_values=k_values,
                    )
                    report_progress(
                        "retrieval_metrics",
                        "complete",
                        (
                            f"Calculated metrics for {query_status}."
                            if relevance["status"] == "available"
                            else f"Metrics unavailable for {query_status}: "
                            f"{relevance['reason']}"
                        ),
                    )

                    question_report: dict[str, Any] = {
                        "question": question,
                        "retrieved_chunk_ids": [item.get("id") for item in retrieved],
                        "retrieval_metrics": retrieval_scores,
                        "retrieval_evaluation": {
                            "status": relevance["status"],
                            "reason": relevance["reason"],
                            "matched_evidence_count": relevance[
                                "matched_evidence_count"
                            ],
                            "unmatched_evidence": relevance[
                                "unmatched_evidence"
                            ],
                            "relevant_chunk_ids": relevance[
                                "relevant_chunk_ids"
                            ],
                        },
                        "retrieval_latency_seconds": retrieval_latency,
                    }

                    report_progress(
                        "answer_generation",
                        "running",
                        f"Generating a grounded answer for {query_status}.",
                    )
                    answer_started = perf_counter()
                    answer = generate_answer(
                        question=question,
                        retrieved_chunks=retrieved,
                        api_key=groq_api_key,
                    )
                    question_report["answer_generation_latency_seconds"] = (
                        perf_counter() - answer_started
                    )
                    report_progress(
                        "answer_generation",
                        "complete",
                        f"Generated the answer for {query_status}.",
                    )

                    report_progress(
                        "response_evaluation",
                        "running",
                        f"Evaluating the generated answer for {query_status}.",
                    )
                    evaluation_started = perf_counter()
                    response_scores = evaluate_response(
                        question=question,
                        answer=answer,
                        retrieved_chunks=retrieved,
                        reference_answer=test_case.get("reference_answer"),
                        expected_key_points=test_case.get("expected_key_points"),
                        citation_sources=test_case.get("citation_sources"),
                        api_key=groq_api_key,
                    )
                    question_report["answer"] = answer
                    question_report["response_metrics"] = response_scores
                    question_report["response_evaluation_latency_seconds"] = (
                        perf_counter() - evaluation_started
                    )
                    question_report["end_to_end_latency_seconds"] = (
                        perf_counter() - pipeline_started
                    )
                    report_progress(
                        "response_evaluation",
                        "complete",
                        f"Evaluated the generated answer for {query_status}.",
                    )

                    question_reports.append(question_report)

                evaluation_summary = _retrieval_evaluation_summary(
                    question_reports
                )
                pipeline_reports.append({
                    "config_id": config_id,
                    "collection_name": collection_name,
                    "chunking_strategy": strategy,
                    "embedding_model": embedding_model,
                    "retrieval_method": retrieval_method,
                    "reranker_enabled": use_reranker,
                    "number_of_searchable_chunks": len(searchable_chunks),
                    "number_of_parent_chunks": len(parent_chunks),
                    "mean_retrieval_metrics": _mean_metrics(
                        question_reports,
                        "retrieval_metrics",
                    ),
                    **evaluation_summary,
                    "mean_retrieval_latency_seconds": mean(
                        row["retrieval_latency_seconds"]
                        for row in question_reports
                    ),
                    "mean_end_to_end_latency_seconds": mean(
                        row["end_to_end_latency_seconds"]
                        for row in question_reports
                    ),
                    "mean_answer_generation_latency_seconds": mean(
                        row["answer_generation_latency_seconds"]
                        for row in question_reports
                    ),
                    "mean_response_evaluation_latency_seconds": mean(
                        row["response_evaluation_latency_seconds"]
                        for row in question_reports
                    ),
                    "mean_response_metrics": _mean_metrics(
                        question_reports,
                        "response_metrics",
                    ),
                    "per_question": question_reports,
                })

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tenant_id": tenant_id,
        "settings": {
            "chunking_strategies": list(strategies),
            "embedding_models": list(models),
            "retrieval_methods": list(methods),
            "retrieval_ground_truth": "evidence_quotes_mapped_to_chunk_ids",
            "top_k": top_k,
            "candidate_k": candidate_k,
            "reranker_enabled": use_reranker,
            "response_evaluation_mandatory": True,
            "number_of_questions": len(cases),
            "k_values": list(k_values),
        },
        "pipelines": pipeline_reports,
        "notes": [
            "Evidence quotes are matched to generated chunk text independently for each chunking strategy; users do not need to supply chunk IDs.",
            "Retrieval metrics are unavailable when evidence is missing or does not match the generated chunks.",
            "HyDE and query expansion require a Groq API key and make additional LLM calls.",
            "Answer generation and response evaluation always run and require Groq, increasing cost and latency.",
        ],
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        report["saved_to"] = str(path)

    return report
