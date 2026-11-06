import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from groq import BadRequestError

from evaluation.response_metrics import evaluate_response
from generation.groq_client import GroqStructuredOutputError


class ResponseMetricsTests(unittest.TestCase):
    def test_judge_uses_compact_json_and_low_gpt_oss_reasoning(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "faithfulness": 1.0,
                                "answer_relevance": 1.0,
                                "correctness": 1.0,
                                "completeness": 1.0,
                                "unsupported_claim_rate": 0.0,
                                "citation_accuracy": None,
                            }
                        )
                    )
                )
            ]
        )

        with patch(
            "evaluation.response_metrics.create_chat_completion",
            return_value=response,
        ) as create_completion:
            scores = evaluate_response(
                question="Where do employees submit expenses?",
                answer="Submit expenses in the finance portal.",
                retrieved_chunks=[
                    {
                        "id": "chunk-1",
                        "text": "Submit expenses in the finance portal.",
                        "metadata": {"source": "handbook.md"},
                    }
                ],
                reference_answer="Submit expenses in the finance portal.",
                api_key="test-key",
            )

        self.assertEqual(scores["faithfulness"], 1.0)
        self.assertEqual(scores["citation_accuracy"], None)
        self.assertEqual(scores["unsupported_claim_rate"], 0.0)
        self.assertEqual(scores["hallucination_rate"], 0.0)
        self.assertTrue(
            all(
                value == "Not provided."
                for value in scores["reasoning"].values()
            )
        )
        request = create_completion.call_args.kwargs
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertEqual(request["reasoning_format"], "hidden")
        self.assertEqual(request["max_completion_tokens"], 700)
        self.assertEqual(request["response_format"], {"type": "json_object"})

    def test_json_validate_failure_returns_actionable_error(self):
        request = httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
        )
        response = httpx.Response(
            400,
            request=request,
            json={
                "error": {
                    "message": "Failed to generate JSON.",
                    "code": "json_validate_failed",
                }
            },
        )
        api_error = BadRequestError(
            "invalid JSON response",
            response=response,
            body={
                "error": {
                    "message": "Failed to generate JSON.",
                    "code": "json_validate_failed",
                }
            },
        )

        with patch(
            "evaluation.response_metrics.create_chat_completion",
            side_effect=api_error,
        ):
            with self.assertRaises(GroqStructuredOutputError) as raised:
                evaluate_response(
                    question="Where do employees submit expenses?",
                    answer="Submit expenses in the finance portal.",
                    retrieved_chunks=[],
                    api_key="test-key",
                )

        self.assertIn("fewer questions per run", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
