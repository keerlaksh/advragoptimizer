from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.common import document_unit_id


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split documents into fixed-size character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
        strip_whitespace=True,
    )

    chunks = []

    for document in documents:
        split_documents = splitter.split_documents([document])
        unit_id = document_unit_id(document.metadata)

        for chunk_index, split_document in enumerate(split_documents):
            metadata = dict(split_document.metadata)

            metadata.update({
                "chunking_strategy": "fixed_size",
                "chunk_index": chunk_index,
                "chunk_id": f"{unit_id}:fixed:{chunk_index}",
                "chunk_size_chars": len(split_document.page_content),
            })

            chunks.append(
                Document(
                    page_content=split_document.page_content,
                    metadata=metadata,
                )
            )

    return chunks