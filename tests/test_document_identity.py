import unittest

from langchain_core.documents import Document

from chunking.fixed_size import chunk_documents as fixed_size_chunks
from chunking.hierarchical import chunk_documents as hierarchical_chunks
from chunking.semantic import chunk_documents as semantic_chunks
from chunking.structure_aware import chunk_documents as structure_aware_chunks
from ingestion.document_loader import load_uploaded_file


class UploadedText:
    name = "policy.txt"

    def __init__(self, content):
        self.content = content

    def getvalue(self):
        return self.content


class DocumentIdentityTests(unittest.TestCase):
    def test_document_id_is_stable_for_same_tenant_file_and_content(self):
        first = load_uploaded_file(
            UploadedText(b"Stable policy content."),
            tenant_id="tenant-a",
        )[0]
        repeated = load_uploaded_file(
            UploadedText(b"Stable policy content."),
            tenant_id="tenant-a",
        )[0]
        changed_content = load_uploaded_file(
            UploadedText(b"Updated policy content."),
            tenant_id="tenant-a",
        )[0]
        other_tenant = load_uploaded_file(
            UploadedText(b"Stable policy content."),
            tenant_id="tenant-b",
        )[0]

        self.assertEqual(
            first.metadata["document_id"],
            repeated.metadata["document_id"],
        )
        self.assertNotEqual(
            first.metadata["document_id"],
            changed_content.metadata["document_id"],
        )
        self.assertNotEqual(
            first.metadata["document_id"],
            other_tenant.metadata["document_id"],
        )

    def test_chunk_ids_are_stable_across_document_order_and_pdf_pages(self):
        page_one = Document(
            page_content="Page one contains a lengthy synthetic policy passage.",
            metadata={
                "document_id": "shared-document",
                "source": "policy.pdf",
                "page": 1,
            },
        )
        page_two = Document(
            page_content="Page two contains another lengthy synthetic passage.",
            metadata={
                "document_id": "shared-document",
                "source": "policy.pdf",
                "page": 2,
            },
        )

        chunkers = (
            lambda documents: fixed_size_chunks(
                documents,
                chunk_size=24,
                chunk_overlap=0,
            ),
            lambda documents: semantic_chunks(
                documents,
                max_chunk_size=24,
            ),
            lambda documents: hierarchical_chunks(
                documents,
                parent_chunk_size=30,
                parent_overlap=0,
                child_chunk_size=16,
                child_overlap=0,
            ),
            lambda documents: structure_aware_chunks(
                documents,
                chunk_size=24,
                chunk_overlap=0,
            ),
        )

        for chunk_documents in chunkers:
            with self.subTest(chunker=chunk_documents):
                forward = chunk_documents([page_one, page_two])
                reverse = chunk_documents([page_two, page_one])
                if hasattr(forward, "child_chunks"):
                    forward_items = forward.parent_chunks + forward.child_chunks
                    reverse_items = reverse.parent_chunks + reverse.child_chunks
                else:
                    forward_items = forward
                    reverse_items = reverse

                forward_ids = {
                    item.metadata["chunk_id"] for item in forward_items
                }
                reverse_ids = {
                    item.metadata["chunk_id"] for item in reverse_items
                }
                self.assertEqual(len(forward_ids), len(forward_items))
                self.assertEqual(forward_ids, reverse_ids)


if __name__ == "__main__":
    unittest.main()
