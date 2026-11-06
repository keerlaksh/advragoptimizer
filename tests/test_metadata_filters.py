import unittest

from retrieval.hybrid_search import build_where_filter
from retrieval.hyde import _build_where_filter
from retrieval.metadata_filters import build_metadata_where


class MetadataFilterTests(unittest.TestCase):
    def test_hierarchical_filter_always_includes_tenant_and_child_level(self):
        expected = {
            "$and": [
                {"tenant_id": "tenant-a"},
                {"chunk_level": "child"},
                {"source": "handbook.pdf"},
            ]
        }

        self.assertEqual(
            build_metadata_where(
                "tenant-a",
                "hierarchical",
                {"source": "handbook.pdf"},
            ),
            expected,
        )
        self.assertEqual(
            build_where_filter(
                "tenant-a",
                "hierarchical",
                {"source": "handbook.pdf"},
            ),
            expected,
        )
        self.assertEqual(
            _build_where_filter(
                "tenant-a",
                "hierarchical",
                {"source": "handbook.pdf"},
            ),
            expected,
        )

    def test_every_retrieval_filter_path_rejects_protected_fields(self):
        builders = (
            build_metadata_where,
            build_where_filter,
            _build_where_filter,
        )

        for build_filter in builders:
            with self.subTest(builder=build_filter.__module__):
                with self.assertRaises(ValueError):
                    build_filter(
                        "tenant-a",
                        "fixed_size",
                        {"tenant_id": "tenant-b"},
                    )


if __name__ == "__main__":
    unittest.main()
