import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.common import document_unit_id


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _split_sentences(text: str) -> list[str]:
    """Split text into approximate sentences while preserving lists/tables."""

    sentences = []

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()

        if not block:
            continue

        lines = [line.strip() for line in block.splitlines() if line.strip()]

        structured = len(lines) > 1 and any(
            "|" in line
            or re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)
            or line.startswith("#")
            for line in lines
        )

        if structured:
            sentences.extend(lines)
        else:
            normalized = " ".join(lines)

            parts = re.split(
                r"(?<=[.!?])\s+",
                normalized,
            )

            sentences.extend(
                part.strip()
                for part in parts
                if part.strip()
            )

    return sentences


def _embed_sentences(
    sentences: list[str],
    embedding_model: Any = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Generate normalized sentence embeddings."""

    if embedding_model is not None:
        if hasattr(embedding_model, "embed_documents"):
            vectors = embedding_model.embed_documents(sentences)
        elif hasattr(embedding_model, "encode"):
            vectors = embedding_model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            raise TypeError(
                "embedding_model must support embed_documents() or encode()."
            )
    else:
        model = _load_model(model_name)

        vectors = model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(sentences):
        raise ValueError("Unexpected sentence embedding shape.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / np.maximum(norms, 1e-12)


def _group_sentences(
    sentences: list[str],
    distances: np.ndarray,
    breakpoint_percentile: float,
    max_chunk_size: int,
) -> list[str]:
    """Group semantically similar sentences into chunks."""

    break_after = np.zeros(len(sentences) - 1, dtype=bool)

    if distances.size and np.ptp(distances) > 1e-8:
        threshold = np.percentile(distances, breakpoint_percentile)
        break_after = distances >= threshold

    groups = []
    current = []

    for index, sentence in enumerate(sentences):
        if current:
            current_length = len(" ".join(current))
            candidate_length = current_length + 1 + len(sentence)

            semantic_boundary = break_after[index - 1]

            size_boundary = candidate_length > max_chunk_size

            if semantic_boundary or size_boundary:
                groups.append(" ".join(current))
                current = []

        current.append(sentence)

    if current:
        groups.append(" ".join(current))

    # Split any individual sentence/group that exceeds the maximum size.
    hard_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        separators=["\n", " ", ""],
    )

    final_groups = []

    for group in groups:
        final_groups.extend(hard_splitter.split_text(group))

    return final_groups


def chunk_documents(
    documents: Sequence[Document],
    breakpoint_percentile: float = 80.0,
    max_chunk_size: int = 1200,
    embedding_model: Any = None,
) -> list[Document]:
    """
    Split documents using semantic changes between adjacent sentences.

    By default, a fixed MiniLM model is used only for segmentation.
    Retrieval embeddings are generated separately later.
    """

    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError("breakpoint_percentile must be between 0 and 100.")

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive.")

    chunks = []

    for document in documents:
        sentences = _split_sentences(document.page_content)

        if not sentences:
            continue

        if len(sentences) == 1:
            groups = [sentences[0]]
            if len(groups[0]) > max_chunk_size:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chunk_size,
                    chunk_overlap=0,
                )
                groups = splitter.split_text(groups[0])
        else:
            vectors = _embed_sentences(
                sentences,
                embedding_model=embedding_model,
            )

            # With normalized vectors, dot product gives cosine similarity.
            similarities = np.sum(
                vectors[:-1] * vectors[1:],
                axis=1,
            )

            distances = 1.0 - np.clip(similarities, -1.0, 1.0)

            groups = _group_sentences(
                sentences=sentences,
                distances=distances,
                breakpoint_percentile=breakpoint_percentile,
                max_chunk_size=max_chunk_size,
            )

        unit_id = document_unit_id(document.metadata)
        document_chunk_index = 0

        for group in groups:
            if not group.strip():
                continue

            metadata = dict(document.metadata)
            chunk_index = document_chunk_index
            document_chunk_index += 1

            metadata.update({
                "chunking_strategy": "semantic",
                "chunk_index": chunk_index,
                "chunk_id": f"{unit_id}:semantic:{chunk_index}",
                "chunk_size_chars": len(group),
            })

            chunks.append(
                Document(
                    page_content=group,
                    metadata=metadata,
                )
            )

    return chunks