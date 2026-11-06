import unittest

from experiments.pipeline_recommender import _score_retrieval, recommend_pipeline


class PipelineRecommenderTests(unittest.TestCase):
    def test_scores_retrieval_from_labelled_metrics(self):
        result = _score_retrieval(
            {
                "hit_rate@5": 1.0,
                "recall@5": 1.0,
                "mrr": 1.0,
                "ndcg@5": 1.0,
                "precision@5": 1.0,
            },
            {
                "hit_rate": 0.15,
                "recall": 0.30,
                "mrr": 0.20,
                "ndcg": 0.25,
                "precision": 0.10,
            },
        )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["source"], "evidence_mapped_chunk_metrics")

    def test_does_not_invent_retrieval_score_without_ground_truth(self):
        result = _score_retrieval(
            {},
            {
                "hit_rate": 0.15,
                "recall": 0.30,
                "mrr": 0.20,
                "ndcg": 0.25,
                "precision": 0.10,
            },
        )

        self.assertIsNone(result["score"])
        self.assertIsNone(result["source"])
        self.assertEqual(result["metrics_used"], {})

    def test_pipeline_ranking_marks_incomplete_comparison(self):
        recommendation = recommend_pipeline(
            {
                "pipelines": [
                    {
                        "config_id": "fixed_minilm",
                        "chunking_strategy": "fixed_size",
                        "embedding_model": "minilm",
                        "retrieval_method": "dense",
                        "mean_retrieval_metrics": {
                            "hit_rate@5": 1.0,
                            "precision@5": 0.2,
                            "recall@5": 1.0,
                            "mrr": 1.0,
                            "ndcg@5": 1.0,
                        },
                        "mean_response_metrics": {
                            "faithfulness": 0.9,
                        },
                    }
                ]
            }
        )

        winner = recommendation["recommended_pipeline"]
        self.assertTrue(winner["comparison_complete"])
        self.assertEqual(
            winner["retrieval_score_source"],
            "evidence_mapped_chunk_metrics",
        )
        self.assertIsNotNone(winner["response_score"])

    def test_response_only_ranking_is_marked_incomplete(self):
        recommendation = recommend_pipeline(
            {
                "pipelines": [
                    {
                        "config_id": "fixed_minilm",
                        "chunking_strategy": "fixed_size",
                        "embedding_model": "minilm",
                        "retrieval_method": "dense",
                        "mean_response_metrics": {
                            "faithfulness": 0.9,
                        },
                    }
                ]
            }
        )

        winner = recommendation["recommended_pipeline"]
        self.assertFalse(winner["comparison_complete"])
        self.assertIsNone(winner["retrieval_score"])
        self.assertTrue(
            any("not a full pipeline comparison" in note
                for note in recommendation["notes"])
        )


if __name__ == "__main__":
    unittest.main()
