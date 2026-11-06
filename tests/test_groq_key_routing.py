import unittest
from unittest.mock import patch

from experiments.experiment_runner import _retrieve


class GroqKeyRoutingTests(unittest.TestCase):
    def test_api_key_is_passed_to_hyde_only(self):
        api_key = "gsk-hyde-test-key"
        common = {
            "question": "question",
            "chroma_manager": object(),
            "chunking_strategy": "fixed_size",
            "embedding_model": "minilm",
            "tenant_id": "tenant-a",
            "top_k": 2,
            "candidate_k": 2,
            "filters": None,
            "use_reranker": False,
            "reranker_model": None,
            "groq_api_key": api_key,
        }

        with patch(
            "experiments.experiment_runner.hyde_search",
            return_value=[],
        ) as hyde:
            _retrieve(method="hyde", **common)

        self.assertEqual(hyde.call_args.kwargs["api_key"], api_key)

    def test_api_key_is_passed_to_query_expansion_only(self):
        api_key = "gsk-expansion-test-key"
        common = {
            "question": "question",
            "chroma_manager": object(),
            "chunking_strategy": "fixed_size",
            "embedding_model": "minilm",
            "tenant_id": "tenant-a",
            "top_k": 2,
            "candidate_k": 2,
            "filters": None,
            "use_reranker": False,
            "reranker_model": None,
            "groq_api_key": api_key,
        }

        with patch(
            "experiments.experiment_runner.query_expansion_search",
            return_value=[],
        ) as query_expansion:
            _retrieve(method="query_expansion", **common)

        self.assertEqual(
            query_expansion.call_args.kwargs["api_key"],
            api_key,
        )


if __name__ == "__main__":
    unittest.main()
