import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.experiment_runner import (
    _retrieval_evaluation_summary,
    run_experiments,
)
from ingestion.document_loader import load_uploaded_file
from vectorstore.chroma_manager import ChromaManager


class UploadedText:
    name = "handbook.txt"

    def getvalue(self):
        return (
            b"Employees can submit a reimbursement request through the "
            b"finance portal. Include receipts with every request."
        )


class TestEmbeddings:
    models = {
        "minilm": {
            "model_id": "test-embedding-model",
        }
    }

    @staticmethod
    def _embed(text):
        vector = [0.0] * 16
        for token in text.lower().split():
            index = int.from_bytes(
                hashlib.sha256(token.encode("utf-8")).digest()[:2],
                "big",
            ) % len(vector)
            vector[index] += 1.0
        return vector

    def embed_documents(self, texts, model_name):
        return [self._embed(text) for text in texts]

    def embed_query(self, query, model_name):
        return self._embed(query)


class BaselinePipelineTests(unittest.TestCase):
    def test_retrieval_status_summarizes_partial_question_coverage(self):
        summary = _retrieval_evaluation_summary(
            [
                {"retrieval_evaluation": {"status": "available"}},
                {"retrieval_evaluation": {"status": "unavailable"}},
            ]
        )

        self.assertEqual(
            summary,
            {
                "retrieval_evaluation_status": "partially_available",
                "retrieval_evaluation_available_questions": 1,
                "retrieval_evaluation_unavailable_questions": 1,
                "retrieval_evaluation_total_questions": 2,
            },
        )

    def test_retrieval_status_is_unavailable_when_no_questions_have_labels(self):
        summary = _retrieval_evaluation_summary(
            [
                {"retrieval_evaluation": {"status": "unavailable"}},
                {"retrieval_evaluation": {"status": "unavailable"}},
            ]
        )

        self.assertEqual(summary["retrieval_evaluation_status"], "unavailable")
        self.assertEqual(summary["retrieval_evaluation_available_questions"], 0)
        self.assertEqual(summary["retrieval_evaluation_unavailable_questions"], 2)

    def test_fixed_minilm_dense_pipeline_generates_and_evaluates_answer(self):
        test_api_key = "gsk-test-key-not-for-reports"
        documents = load_uploaded_file(
            UploadedText(),
            tenant_id="tenant-a",
        )
        real_chroma_manager = ChromaManager
        managers = []

        def create_chroma_manager(persist_directory):
            manager = real_chroma_manager(
                persist_directory=persist_directory,
                embedding_manager=TestEmbeddings(),
            )
            managers.append(manager)
            return manager

        with tempfile.TemporaryDirectory() as temp_dir:
            progress_events = []
            try:
                with (
                    patch(
                        "experiments.experiment_runner.ChromaManager",
                        side_effect=create_chroma_manager,
                    ),
                    patch(
                        "experiments.experiment_runner.generate_answer",
                        return_value="Submit the request through the finance portal.",
                    ) as generate_answer,
                    patch(
                        "experiments.experiment_runner.evaluate_response",
                        return_value={
                            "faithfulness": 1.0,
                            "answer_relevance": 1.0,
                        },
                    ) as evaluate_response,
                ):
                    report = run_experiments(
                        documents=documents,
                        test_cases=[
                            {
                                "question": (
                                    "Where do employees submit reimbursement "
                                    "requests?"
                                ),
                                "reference_answer": (
                                    "Employees submit requests through the "
                                    "finance portal."
                                ),
                                "evidence": (
                                    "submit a reimbursement request through "
                                    "the finance portal"
                                ),
                            }
                        ],
                        tenant_id="tenant-a",
                        persist_directory=str(Path(temp_dir) / "chroma"),
                        chunking_strategies=["fixed_size"],
                        embedding_models=["minilm"],
                        retrieval_methods=["dense"],
                        top_k=2,
                        candidate_k=2,
                        groq_api_key=test_api_key,
                        progress_callback=lambda stage, state, message: (
                            progress_events.append((stage, state, message))
                        ),
                    )
            finally:
                for manager in managers:
                    manager.client.close()

        self.assertEqual(len(report["pipelines"]), 1)
        pipeline = report["pipelines"][0]
        question = pipeline["per_question"][0]
        self.assertEqual(pipeline["chunking_strategy"], "fixed_size")
        self.assertEqual(pipeline["embedding_model"], "minilm")
        self.assertEqual(pipeline["retrieval_method"], "dense")
        self.assertTrue(question["retrieved_chunk_ids"])
        self.assertEqual(
            question["answer"],
            "Submit the request through the finance portal.",
        )
        self.assertGreaterEqual(question["retrieval_latency_seconds"], 0)
        self.assertGreaterEqual(question["end_to_end_latency_seconds"], 0)
        self.assertIn("mean_retrieval_latency_seconds", pipeline)
        self.assertEqual(
            pipeline["mean_retrieval_metrics"]["hit_rate@5"],
            1.0,
        )
        self.assertEqual(
            pipeline["mean_retrieval_metrics"]["recall@5"],
            1.0,
        )
        self.assertEqual(question["retrieval_evaluation"]["status"], "available")
        self.assertEqual(
            question["retrieval_evaluation"]["relevant_chunk_ids"],
            question["retrieved_chunk_ids"],
        )
        self.assertIn("mean_response_metrics", pipeline)
        generate_answer.assert_called_once()
        evaluate_response.assert_called_once()
        self.assertEqual(
            generate_answer.call_args.kwargs["api_key"],
            test_api_key,
        )
        self.assertEqual(
            evaluate_response.call_args.kwargs["api_key"],
            test_api_key,
        )
        self.assertNotIn(test_api_key, str(report))
        completed_stages = {
            stage
            for stage, state, _ in progress_events
            if state == "complete"
        }
        self.assertTrue(
            {
                "chunking",
                "embedding_storage",
                "retrieval",
                "retrieval_metrics",
                "answer_generation",
                "response_evaluation",
            }.issubset(completed_stages)
        )

    def test_dense_and_hybrid_runs_always_generate_and_evaluate_answers(self):
        documents = load_uploaded_file(
            UploadedText(),
            tenant_id="tenant-a",
        )
        real_chroma_manager = ChromaManager
        managers = []

        def create_chroma_manager(persist_directory):
            manager = real_chroma_manager(
                persist_directory=persist_directory,
                embedding_manager=TestEmbeddings(),
            )
            managers.append(manager)
            return manager

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with (
                    patch(
                        "experiments.experiment_runner.ChromaManager",
                        side_effect=create_chroma_manager,
                    ),
                    patch(
                        "experiments.experiment_runner.generate_answer",
                        return_value="Submit the request through the finance portal.",
                    ) as generate_answer,
                    patch(
                        "experiments.experiment_runner.evaluate_response",
                        return_value={"faithfulness": 1.0, "correctness": 1.0},
                    ) as evaluate_response,
                ):
                    report = run_experiments(
                        documents=documents,
                        test_cases=[
                            {
                                "question": "Where submit reimbursement requests?",
                                "reference_answer": "Through the finance portal.",
                                "evidence": "through the finance portal",
                            }
                        ],
                        tenant_id="tenant-a",
                        persist_directory=str(Path(temp_dir) / "chroma"),
                        chunking_strategies=["fixed_size"],
                        embedding_models=["minilm"],
                        retrieval_methods=["dense", "hybrid"],
                        top_k=2,
                        candidate_k=2,
                        groq_api_key="gsk-test-key-not-for-reports",
                    )
            finally:
                for manager in managers:
                    manager.client.close()

        self.assertEqual(len(report["pipelines"]), 2)
        self.assertEqual(generate_answer.call_count, 2)
        self.assertEqual(evaluate_response.call_count, 2)
        for pipeline in report["pipelines"]:
            self.assertEqual(
                pipeline["retrieval_evaluation_status"],
                "available",
            )
            self.assertEqual(
                pipeline["mean_retrieval_metrics"]["hit_rate@5"],
                1.0,
            )
            self.assertEqual(
                pipeline["mean_response_metrics"]["faithfulness"],
                1.0,
            )

    def test_experiments_require_groq_for_mandatory_response_evaluation(self):
        documents = load_uploaded_file(
            UploadedText(),
            tenant_id="tenant-a",
        )
        with self.assertRaisesRegex(ValueError, "Groq API key is required"):
            run_experiments(
                documents=documents,
                test_cases=[{"question": "Where are requests submitted?"}],
                tenant_id="tenant-a",
                chunking_strategies=["fixed_size"],
                embedding_models=["minilm"],
                retrieval_methods=["dense"],
                groq_api_key=None,
            )


if __name__ == "__main__":
    unittest.main()
