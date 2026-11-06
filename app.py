from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import GROQ_MODEL
from generation.groq_client import (
    GroqRateLimitError,
    GroqStructuredOutputError,
)
from ingestion.document_loader import load_uploaded_file
from experiments.experiment_runner import (
    CHUNKING_STRATEGIES,
    EMBEDDING_MODELS,
    RETRIEVAL_METHODS,
    run_experiments,
)
from experiments.pipeline_recommender import (
    recommend_pipeline,
    save_recommendation,
)


APP_TITLE = "advrag | RAG Pipeline Optimizer"
REPORT_DIR = Path("experiments") / "results"
REPORT_PATH = REPORT_DIR / "experiment_report.json"
RECOMMENDATION_PATH = REPORT_DIR / "pipeline_recommendation.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 advrag — RAG Pipeline Optimizer")
st.caption(
    "Compare chunking, embedding, and retrieval configurations using benchmark questions and source evidence."
)

with st.sidebar:
    st.header("Experiment settings")
    st.warning(
        "Development/demo mode: the tenant ID below is manually entered. "
        "In production, derive it from authenticated user permissions on the server; "
        "never let an end user choose another tenant ID."
    )
    tenant_id = st.text_input("Tenant ID (development only)", value="company_A")
    groq_api_key = st.text_input(
        "Groq API key",
        type="password",
        key="groq_api_key_input",
        help=(
            "Used automatically by HyDE/query expansion when selected, and "
            "always for answer generation and response evaluation. It is not "
            "included in reports."
        ),
    ).strip()
    st.caption(f"Groq model: {GROQ_MODEL}")
    persist_directory = st.text_input(
        "ChromaDB storage directory", value="chroma_data"
    )

    st.subheader("Chunking strategies")
    selected_chunking = st.multiselect(
        "Strategies to compare",
        options=list(CHUNKING_STRATEGIES),
        default=list(CHUNKING_STRATEGIES),
        key="chunking_strategies_v2",
        help="The default compares all four chunking strategies.",
    )

    st.subheader("Embedding models")
    selected_embeddings = st.multiselect(
        "Models to compare",
        options=list(EMBEDDING_MODELS),
        default=list(EMBEDDING_MODELS),
        key="embedding_models_v2",
    )

    st.subheader("Retrieval methods")
    selected_retrieval = st.multiselect(
        "Methods to compare",
        options=list(RETRIEVAL_METHODS),
        default=["dense", "hybrid"],
        key="retrieval_methods_v2",
        help=(
            "Dense and Hybrid do not use a generative LLM. HyDE and Query "
            "Expansion use Groq to transform the query."
        ),
    )

    top_k = st.number_input("Final chunks (top_k)", min_value=1, max_value=50, value=5)
    candidate_k = st.number_input(
        "Initial candidates (candidate_k)",
        min_value=int(top_k),
        max_value=200,
        value=max(20, int(top_k)),
        help="The reranker, if enabled, reranks this candidate set before returning top_k chunks.",
    )
    use_reranker = st.checkbox("Enable cross-encoder reranking", value=False)
    reranker_model = st.text_input(
        "Reranker model",
        value="cross-encoder/ms-marco-MiniLM-L6-v2",
        disabled=not use_reranker,
    )

    st.subheader("Evaluation")
    st.caption(
        "Dense, Hybrid, reranking, and retrieval metrics run without Groq. "
        "HyDE and Query Expansion use Groq automatically when selected. "
        "Every run also uses Groq for mandatory answer generation and response "
        "evaluation. "
        "To calculate retrieval metrics, add verified source evidence quotes "
        "to each benchmark question; the app maps quotes to its generated chunk IDs."
    )
    max_questions = st.number_input(
        "Maximum test questions (0 = all)",
        min_value=0,
        max_value=500,
        value=3,
    )

st.header("1. Upload company documents")
uploaded_files = st.file_uploader(
    "Upload one or more PDF, DOCX, TXT, or Markdown documents",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
    help="Documents will be assigned the development tenant ID above for this experiment.",
)

st.header("2. Provide your evaluation dataset")
st.write(
    "Provide benchmark questions and verified reference answers. Add a supporting "
    "source quote for mathematical retrieval metrics. The app maps "
    "quotes to chunk IDs internally for each chunking strategy."
)

with st.expander("Evaluation dataset JSON format", expanded=True):
    st.markdown(
        "Each case needs a `question`. Add a verified `reference_answer` for "
        "response evaluation and an `evidence` quote copied from the uploaded "
        "source to enable retrieval metrics. For multiple supporting passages, "
        "use `evidence_passages` (a list of quotes). The app maps quotes to the "
        "generated chunk IDs separately for each strategy; users never provide "
        "chunk IDs. If evidence is missing or cannot be matched exactly after "
        "whitespace normalization, retrieval metrics are marked unavailable."
    )
    st.code(
        '''[
  {
    "question": "How many days of annual leave do employees receive?",
    "reference_answer": "Employees receive 20 days of annual paid leave.",
    "evidence": "Employees are entitled to 20 days of annual paid leave."
  }
]''',
        language="json",
    )

uploaded_test_file = st.file_uploader(
    "Optional: upload evaluation dataset (.json)",
    type=["json"],
    key="test_dataset_file",
)
manual_dataset_text = st.text_area(
    "Or paste the evaluation dataset JSON here",
    value="",
    height=180,
    placeholder='[\n  {"question": "A question answerable from the uploaded document", "reference_answer": "A verified answer from the document"}\n]',
)


def parse_test_cases() -> list[dict[str, Any]]:
    """Parse either the uploaded JSON file or the JSON text area."""
    if uploaded_test_file is not None:
        raw = uploaded_test_file.getvalue().decode("utf-8-sig")
    else:
        raw = manual_dataset_text.strip()

    if not raw:
        raise ValueError("Upload or paste a JSON evaluation dataset first.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, list) or not data:
        raise ValueError("The evaluation dataset must be a non-empty JSON list.")

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Test case {index} must be a JSON object.")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Test case {index} needs a non-empty 'question'.")
        for field in ("expected_key_points", "evidence_passages"):
            if field in item and not isinstance(item[field], list):
                raise ValueError(f"Test case {index}: '{field}' must be a list.")
        if "evidence" in item and not isinstance(item["evidence"], str):
            raise ValueError(f"Test case {index}: 'evidence' must be a string.")
        for field in (
            "relevant_chunk_ids_by_strategy",
            "relevant_chunk_ids_by_config",
            "filters",
        ):
            if field in item and not isinstance(item[field], dict):
                raise ValueError(f"Test case {index}: '{field}' must be an object.")

    return data


run_clicked = st.button("🚀 Run RAG experiments", type="primary", use_container_width=True)

if run_clicked:
    # Never leave results from an earlier run visible when the new run fails.
    st.session_state.pop("advrag_last_report", None)
    st.session_state.pop("advrag_last_recommendation", None)
    if not uploaded_files:
        st.error("Upload at least one company document first.")
    elif not tenant_id.strip():
        st.error("Tenant ID cannot be empty.")
    elif not selected_chunking:
        st.error("Select at least one chunking strategy.")
    elif not selected_embeddings:
        st.error("Select at least one embedding model.")
    elif not selected_retrieval:
        st.error("Select at least one retrieval method.")
    elif not groq_api_key and not os.getenv("GROQ_API_KEY"):
        st.error(
            "A Groq API key is required for mandatory answer generation and "
            "response evaluation. Enter it in the sidebar or configure "
            "GROQ_API_KEY in the environment."
        )
    else:
        stage_labels = {
            "evaluation_dataset": "Evaluation dataset",
            "ingestion": "Document ingestion",
            "chunking": "Chunking",
            "embedding_storage": "Embedding + ChromaDB storage",
            "retrieval": "Retrieval",
            "retrieval_metrics": "Retrieval metrics",
            "answer_generation": "Answer generation",
            "response_evaluation": "Response evaluation",
            "recommendation": "Pipeline recommendation",
        }
        stage_slots = {}
        progress_status = st.status("RAG experiment progress", expanded=True)
        with progress_status:
            for stage, label in stage_labels.items():
                stage_slots[stage] = st.empty()
                stage_slots[stage].markdown(f"◯ **{label}:** Pending")

        active_stage = {"name": None}

        def update_stage(stage: str, state: str, message: str) -> None:
            if state == "running":
                active_stage["name"] = stage
            elif active_stage["name"] == stage:
                active_stage["name"] = None
            icons = {
                "running": "🔄",
                "complete": "✅",
                "failed": "❌",
            }
            stage_slots[stage].markdown(
                f"{icons[state]} **{stage_labels[stage]}:** {message}"
            )

        try:
            update_stage(
                "evaluation_dataset",
                "running",
                "Validating benchmark questions and evidence.",
            )
            test_cases = parse_test_cases()
            max_q = int(max_questions) if int(max_questions) > 0 else None
            update_stage(
                "evaluation_dataset",
                "complete",
                f"Validated {len(test_cases[:max_q] if max_q else test_cases)} question(s).",
            )

            update_stage(
                "ingestion",
                "running",
                f"Reading {len(uploaded_files)} uploaded document(s).",
            )
            source_documents = []
            loaded_document_ids: set[str] = set()
            for uploaded in uploaded_files:
                loaded_documents = load_uploaded_file(
                    uploaded,
                    tenant_id=tenant_id.strip(),
                )
                document_id = loaded_documents[0].metadata["document_id"]
                if document_id in loaded_document_ids:
                    continue
                loaded_document_ids.add(document_id)
                source_documents.extend(loaded_documents)

            if not source_documents:
                raise ValueError("No readable text was extracted from the uploaded documents.")

            update_stage(
                "ingestion",
                "complete",
                f"Loaded {len(source_documents)} text unit(s) from "
                f"{len(uploaded_files)} uploaded file(s).",
            )
            st.info(
                f"Loaded {len(source_documents)} text unit(s) from "
                f"{len(uploaded_files)} uploaded file(s)."
            )

            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            report = run_experiments(
                documents=source_documents,
                test_cases=test_cases,
                tenant_id=tenant_id.strip(),
                persist_directory=persist_directory.strip() or "chroma_data",
                chunking_strategies=selected_chunking,
                embedding_models=selected_embeddings,
                retrieval_methods=selected_retrieval,
                top_k=int(top_k),
                candidate_k=int(candidate_k),
                use_reranker=use_reranker,
                reranker_model=reranker_model.strip() if use_reranker else None,
                groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY"),
                max_questions=max_q,
                output_path=REPORT_PATH,
                progress_callback=update_stage,
            )

            st.session_state["advrag_last_report"] = report
            st.session_state["advrag_last_tenant"] = tenant_id.strip()
            update_stage(
                "recommendation",
                "running",
                "Comparing measured retrieval and response quality.",
            )
            try:
                recommendation = recommend_pipeline(
                    report,
                    require_response_metrics=True,
                )
            except (ValueError, TypeError, KeyError) as exc:
                update_stage("recommendation", "failed", str(exc))
                st.warning(
                    "The experiment completed, but no pipeline could be "
                    "recommended from the available measured metrics."
                )
                st.caption(str(exc))
            else:
                save_recommendation(recommendation, RECOMMENDATION_PATH)
                st.session_state["advrag_last_recommendation"] = recommendation
                update_stage(
                    "recommendation",
                    "complete",
                    f"Ranked {len(recommendation['ranking'])} evaluated pipeline(s).",
                )

            progress_status.update(
                label="RAG experiment workflow finished",
                state="complete",
            )
            st.success(
                f"Experiments completed: {len(report.get('pipelines', []))} pipeline variants evaluated."
            )

        except GroqRateLimitError as exc:
            if active_stage["name"]:
                update_stage(active_stage["name"], "failed", str(exc))
            progress_status.update(
                label="RAG experiment stopped by Groq rate limiting",
                state="error",
            )
            st.warning(str(exc))
        except GroqStructuredOutputError as exc:
            if active_stage["name"]:
                update_stage(active_stage["name"], "failed", str(exc))
            progress_status.update(
                label="RAG experiment stopped by an invalid Groq response",
                state="error",
            )
            st.warning(str(exc))
        except Exception as exc:
            if active_stage["name"]:
                update_stage(active_stage["name"], "failed", str(exc))
            progress_status.update(label="RAG experiment failed", state="error")
            st.exception(exc)

report = st.session_state.get("advrag_last_report")
if report:
    st.divider()
    st.header("3. Experiment results")

    settings = report.get("settings", {})
    metrics_rows = []
    if settings.get("retrieval_ground_truth") != (
        "evidence_quotes_mapped_to_chunk_ids"
    ):
        st.warning(
            "This is a report from an earlier app version. Rerun the experiment "
            "to use source-evidence mapping for retrieval metrics."
        )
    for pipeline in report.get("pipelines", []):
        retrieval_metrics = pipeline.get("mean_retrieval_metrics") or {}
        response_metrics = pipeline.get("mean_response_metrics") or {}
        question_results = pipeline.get("per_question", [])
        available_questions = sum(
            1
            for item in question_results
            if item.get("retrieval_evaluation", {}).get("status") == "available"
        )
        total_questions = len(question_results)
        unavailable_questions = total_questions - available_questions
        has_question_statuses = any(
            isinstance(item.get("retrieval_evaluation"), dict)
            for item in question_results
        )
        retrieval_is_available = (
            available_questions > 0
            if has_question_statuses
            else any(value is not None for value in retrieval_metrics.values())
        )
        if has_question_statuses:
            retrieval_status = (
                "available"
                if total_questions and available_questions == total_questions
                else "partially_available"
                if available_questions
                else "unavailable"
            )
        else:
            retrieval_status = pipeline.get(
                "retrieval_evaluation_status",
                "available" if retrieval_is_available else "unavailable",
            )
        reason_counts: dict[str, int] = {}
        for item in question_results:
            evaluation = item.get("retrieval_evaluation", {})
            reason = evaluation.get("reason")
            if evaluation.get("status") != "available" and reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        unavailable_reasons = "; ".join(
            f"{count}/{total_questions} question(s): {reason}"
            for reason, count in sorted(reason_counts.items())
        )
        if retrieval_status == "partially_available":
            retrieval_status_label = (
                f"Partially available ({available_questions}/{total_questions} "
                "questions)"
            )
        elif retrieval_status == "available":
            retrieval_status_label = (
                f"Available ({available_questions}/{total_questions} questions)"
                if total_questions
                else "Available"
            )
        else:
            retrieval_status_label = (
                f"Unavailable (0/{total_questions} questions)"
                if total_questions
                else "Unavailable"
            )
        metrics_rows.append(
            {
                "Configuration": pipeline.get("config_id"),
                "Retrieval": pipeline.get("retrieval_method"),
                "Reranker": pipeline.get("reranker_enabled", False),
                "Chunking": pipeline.get("chunking_strategy"),
                "Embedding": pipeline.get("embedding_model"),
                "Chunks": pipeline.get("number_of_searchable_chunks"),
                "Retrieval latency (s)": pipeline.get(
                    "mean_retrieval_latency_seconds"
                ),
                "End-to-end latency (s)": pipeline.get(
                    "mean_end_to_end_latency_seconds"
                ),
                "Retrieval score source": (
                    "Evidence quote → internal chunk IDs"
                    if retrieval_is_available
                    else "Unavailable"
                ),
                "Retrieval metrics status": retrieval_status_label,
                "Retrieval evaluation coverage": (
                    f"{available_questions} available / "
                    f"{unavailable_questions} unavailable"
                ),
                "Unavailable question reasons": (
                    unavailable_reasons or "None"
                ),
                "Hit Rate@5": (
                    retrieval_metrics.get("hit_rate@5")
                    if retrieval_is_available
                    else None
                ),
                "Recall@5": (
                    retrieval_metrics.get("recall@5")
                    if retrieval_is_available
                    else None
                ),
                "Precision@5": (
                    retrieval_metrics.get("precision@5")
                    if retrieval_is_available
                    else None
                ),
                "MRR": (
                    retrieval_metrics.get("mrr")
                    if retrieval_is_available
                    else None
                ),
                "nDCG@5": (
                    retrieval_metrics.get("ndcg@5")
                    if retrieval_is_available
                    else None
                ),
                "Faithfulness": response_metrics.get("faithfulness"),
                "Answer relevance": response_metrics.get("answer_relevance"),
                "Correctness": response_metrics.get("correctness"),
                "Completeness": response_metrics.get("completeness"),
                "Hallucination rate (higher = worse)": response_metrics.get(
                    "hallucination_rate",
                    response_metrics.get("unsupported_claim_rate"),
                ),
                "Citation accuracy": response_metrics.get("citation_accuracy"),
            }
        )

    if metrics_rows:
        result_df = pd.DataFrame(metrics_rows)
        result_df = result_df.fillna("N/A")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.caption(
            "Retrieval availability is shown per question: partial means some "
            "questions had verified relevance labels and others did not. "
            "Hallucination rate is the evaluator's unsupported-claim rate "
            "(0 = no unsupported claims; 1 = all claims unsupported)."
        )

    st.download_button(
        "⬇️ Download experiment report (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
        file_name="experiment_report.json",
        mime="application/json",
    )

    st.header("4. Pipeline recommendation")
    try:
        recommendation = st.session_state.get("advrag_last_recommendation")
        if recommendation is None:
            recommendation = recommend_pipeline(
                report,
                require_response_metrics=True,
            )
        winner = recommendation["recommended_pipeline"]
        st.subheader("Best-scoring evaluated pipeline")
        st.write(f"**Pipeline:** {winner['label']}")
        st.metric("Overall score", f"{winner['overall_score']:.3f}")
        left, right = st.columns(2)
        with left:
            score = winner.get("retrieval_score")
            st.metric(
                "Retrieval score",
                f"{score:.3f}" if score is not None else "Not available",
            )
        with right:
            score = winner.get("response_score")
            st.metric(
                "Response score",
                f"{score:.3f}" if score is not None else "Not available",
            )

        st.caption(
            "This is the best-scoring configuration among the tested variants, "
            "not a guarantee of universal performance. Verify labels and inspect failures."
        )
        if not winner.get("comparison_complete"):
            st.warning(
                "This recommendation is provisional: one quality group is "
                "unavailable. Add matching source evidence quotes to the "
                "evaluation dataset for a retrieval-and-response comparison."
            )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rank": item["rank"],
                        "Pipeline": item["label"],
                        "Overall score": item["overall_score"],
                        "Retrieval score": item["retrieval_score"],
                        "Retrieval score source": item[
                            "retrieval_score_source"
                        ],
                        "Response score": item["response_score"],
                        "Both quality groups evaluated": item[
                            "comparison_complete"
                        ],
                    }
                    for item in recommendation["ranking"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        if recommendation.get("notes"):
            with st.expander("Recommendation notes"):
                for note in recommendation["notes"]:
                    st.write(f"- {note}")

        save_recommendation(recommendation, RECOMMENDATION_PATH)
        st.download_button(
            "⬇️ Download pipeline recommendation (JSON)",
            data=json.dumps(recommendation, ensure_ascii=False, indent=2, default=str),
            file_name="pipeline_recommendation.json",
            mime="application/json",
        )

    except (ValueError, TypeError, KeyError) as exc:
        st.warning(
            "A pipeline recommendation could not be calculated. Add matching "
            "source evidence quotes and verified reference answers so both "
            "retrieval and response quality can be measured."
        )
        st.caption(str(exc))

    with st.expander("Inspect detailed per-question results"):
        for pipeline in report.get("pipelines", []):
            label = (
                f"{pipeline.get('chunking_strategy')} + "
                f"{pipeline.get('embedding_model')} + "
                f"{pipeline.get('retrieval_method')}"
            )
            with st.expander(label):
                st.json(pipeline.get("per_question", []))

    with st.expander("View complete experiment report"):
        st.json(report)

st.divider()
st.caption(
    "Prototype note: This UI is for development and evaluation. Before production, "
    "add authenticated admin/user roles, trusted tenant assignment, document-level "
    "authorization, upload limits, and appropriate data-retention controls."
)
