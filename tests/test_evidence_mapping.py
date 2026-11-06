import unittest

from langchain_core.documents import Document

from evaluation.evidence_mapping import (
    get_evidence_passages,
    map_evidence_to_chunks,
    resolve_relevant_chunks,
)


class EvidenceMappingTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Document(
                page_content="Employees receive 20 days of annual paid leave.",
                metadata={"chunk_id": "fixed:0"},
            ),
            Document(
                page_content="Leave requests must be sent to the HR portal.",
                metadata={"chunk_id": "fixed:1"},
            ),
        ]

    def test_maps_evidence_quote_to_generated_chunk_ids(self):
        result = map_evidence_to_chunks(
            ["employees receive 20 days\nof annual paid leave."],
            self.chunks,
        )

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["relevant_chunk_ids"], ["fixed:0"])
        self.assertEqual(result["matched_evidence_count"], 1)

    def test_does_not_use_partial_matches_as_ground_truth(self):
        result = map_evidence_to_chunks(
            [
                "Employees receive 20 days of annual paid leave.",
                "An unsupported extra evidence statement.",
            ],
            self.chunks,
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["relevant_chunk_ids"], ["fixed:0"])
        self.assertEqual(
            result["unmatched_evidence"],
            ["An unsupported extra evidence statement."],
        )

    def test_missing_evidence_has_clear_unavailable_reason(self):
        result = resolve_relevant_chunks(
            {"question": "How much annual leave?"},
            "fixed_size",
            "fixed_minilm",
            self.chunks,
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("No evidence quote", result["reason"])
        self.assertEqual(result["relevant_chunk_ids"], [])

    def test_supports_multiple_evidence_quotes_without_ids(self):
        result = resolve_relevant_chunks(
            {
                "evidence_passages": [
                    "Employees receive 20 days of annual paid leave.",
                    "Leave requests must be sent to the HR portal.",
                ]
            },
            "fixed_size",
            "fixed_minilm",
            self.chunks,
        )

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["relevant_chunk_ids"], ["fixed:0", "fixed:1"])

    def test_rejects_conflicting_evidence_fields(self):
        with self.assertRaisesRegex(ValueError, "either 'evidence'"):
            get_evidence_passages({
                "evidence": "quote",
                "evidence_passages": ["another quote"],
            })


if __name__ == "__main__":
    unittest.main()
