from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.common import document_unit_id


@dataclass
class HierarchicalChunkingResult:
    """Parent chunks and searchable child chunks."""

    parent_chunks: list[Document]
    child_chunks: list[Document]


def chunk_documents(
    documents: Sequence[Document],
    parent_chunk_size: int = 2000,
    parent_overlap: int = 100,
    child_chunk_size: int = 500,
    child_overlap: int = 75,
) -> HierarchicalChunkingResult:
    """Create larger parent chunks and smaller child chunks."""

    if parent_chunk_size <= 0 or child_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive.")

    if parent_overlap < 0 or parent_overlap >= parent_chunk_size:
        raise ValueError("Invalid parent overlap.")

    if child_overlap < 0 or child_overlap >= child_chunk_size:
        raise ValueError("Invalid child overlap.")

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    parents = []
    children = []

    for document in documents:
        parent_documents = parent_splitter.split_documents([document])
        unit_id = document_unit_id(document.metadata)

        for parent_index, parent_document in enumerate(parent_documents):
            base_metadata = dict(parent_document.metadata)
            parent_id = f"{unit_id}:hierarchical:parent:{parent_index}"

            parent_metadata = {
                **base_metadata,
                "chunking_strategy": "hierarchical",
                "chunk_level": "parent",
                "chunk_index": parent_index,
                "chunk_id": parent_id,
            }

            parent = Document(
                page_content=parent_document.page_content,
                metadata=parent_metadata,
            )

            parents.append(parent)

            child_documents = child_splitter.split_documents([parent])

            for child_index, child_document in enumerate(child_documents):
                child_metadata = dict(child_document.metadata)

                child_id = (
                    f"{parent_id}:child:{child_index}"
                )

                child_metadata.update({
                    "chunking_strategy": "hierarchical",
                    "chunk_level": "child",
                    "chunk_index": child_index,
                    "chunk_id": child_id,
                    "parent_chunk_id": parent_id,
                    "chunk_size_chars": len(child_document.page_content),
                })

                children.append(
                    Document(
                        page_content=child_document.page_content,
                        metadata=child_metadata,
                    )
                )

    return HierarchicalChunkingResult(
        parent_chunks=parents,
        child_chunks=children,
    )