from pathlib import Path
from collections.abc import Sequence
from typing import Any

import chromadb
from langchain_core.documents import Document

from embeddings.embedding_manager import EmbeddingManager
from retrieval.metadata_filters import build_metadata_where


CHUNKING_STRATEGIES = {
    "fixed_size",
    "semantic",
    "hierarchical",
    "structure_aware",
}

EMBEDDING_MODELS = {
    "minilm",
    "e5-small",
    "bge-base",
}


class ChromaManager:
    """Manage persistent ChromaDB collections for RAGOptimizer."""

    def __init__(
        self,
        persist_directory: str = "chroma_data",
        embedding_manager: EmbeddingManager | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Persistent storage keeps collections available after restarts.
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory.resolve())
        )
        self.embedding_manager = embedding_manager or EmbeddingManager()

    @staticmethod
    def _validate_configuration(
        chunking_strategy: str,
        embedding_model: str,
    ) -> None:
        if chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError(
                f"Unknown chunking strategy: {chunking_strategy}. "
                f"Choose from {sorted(CHUNKING_STRATEGIES)}."
            )

        if embedding_model not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown embedding model: {embedding_model}. "
                f"Choose from {sorted(EMBEDDING_MODELS)}."
            )

    @staticmethod
    def get_collection_name(
        chunking_strategy: str,
        embedding_model: str,
    ) -> str:
        """Generate a separate collection name for each configuration."""
        ChromaManager._validate_configuration(
            chunking_strategy,
            embedding_model,
        )
        safe_model_name = embedding_model.replace("-", "_")
        return f"{chunking_strategy}_{safe_model_name}"

    def get_collection(
        self,
        chunking_strategy: str,
        embedding_model: str,
    ):
        """Get or create the Chroma collection for one configuration."""
        self._validate_configuration(chunking_strategy, embedding_model)
        collection_name = self.get_collection_name(
            chunking_strategy,
            embedding_model,
        )
        model_id = self.embedding_manager.models[embedding_model]["model_id"]

        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "chunking_strategy": chunking_strategy,
                "embedding_model": embedding_model,
                "embedding_model_id": model_id,
            },
            configuration={"hnsw": {"space": "cosine"}},
            # We create embeddings ourselves using EmbeddingManager.
            embedding_function=None,
        )

    @staticmethod
    def _prepare_metadata(
        document: Document,
        chunking_strategy: str,
        embedding_model: str,
    ) -> dict[str, Any]:
        """Validate and prepare scalar metadata supported by ChromaDB."""
        metadata = dict(document.metadata)

        tenant_id = metadata.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError(
                "Every chunk must contain a valid tenant_id. "
                "Set it when loading the original document."
            )

        chunk_id = metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                "Every chunk must have a unique non-empty chunk_id."
            )

        metadata.update(
            {
                "chunking_strategy": chunking_strategy,
                "embedding_model": embedding_model,
                "config_id": (
                    f"{chunking_strategy}_{embedding_model.replace('-', '_')}"
                ),
            }
        )

        clean_metadata: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean_metadata[str(key)] = value
            else:
                raise TypeError(
                    f"Unsupported metadata value for '{key}': "
                    f"{type(value).__name__}. Use strings, integers, "
                    "floats, booleans, or omit the field."
                )

        return clean_metadata

    def store_chunks(
        self,
        chunks: Sequence[Document],
        chunking_strategy: str,
        embedding_model: str,
        parent_chunks: Sequence[Document] | None = None,
        replace_existing: bool = True,
        batch_size: int = 128,
    ) -> str:
        """Embed and store searchable chunks (and optional parent chunks).

        For hierarchical chunking, pass child chunks as `chunks` and parent
        chunks as `parent_chunks`. Parent and child records are stored in the
        same configuration collection; search filters return children only.
        """
        self._validate_configuration(chunking_strategy, embedding_model)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not chunks:
            raise ValueError("No searchable chunks were provided.")

        parents = list(parent_chunks or [])
        searchable_chunks = list(chunks)
        records = parents + searchable_chunks

        collection = self.get_collection(chunking_strategy, embedding_model)

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for document in records:
            text = document.page_content.strip()
            if not text:
                continue

            metadata = self._prepare_metadata(
                document,
                chunking_strategy,
                embedding_model,
            )
            chunk_id = metadata["chunk_id"]

            ids.append(chunk_id)
            texts.append(text)
            metadatas.append(metadata)

        if not ids:
            raise ValueError("No non-empty chunks were found.")
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate chunk IDs found. Ensure all parent and child IDs "
                "are unique within this configuration."
            )

        # Generate embeddings with the same model used later for query vectors.
        all_embeddings = self.embedding_manager.embed_documents(
            texts,
            embedding_model,
        )
        if len(all_embeddings) != len(texts):
            raise RuntimeError(
                "The number of embeddings does not match the number of chunks."
            )

        # Replacing a source removes old records for the same tenant and source.
        if replace_existing:
            source_pairs = {
                (metadata["tenant_id"], metadata.get("source"))
                for metadata in metadatas
                if metadata.get("source")
            }
            for tenant_id, source in source_pairs:
                collection.delete(
                    where={
                        "$and": [
                            {"tenant_id": tenant_id},
                            {"source": source},
                        ]
                    }
                )

        # Upsert each batch to avoid sending excessive data in one write.
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=all_embeddings[start:end],
                metadatas=metadatas[start:end],
            )

        return collection.name

    def search(
        self,
        question: str,
        chunking_strategy: str,
        embedding_model: str,
        tenant_id: str,
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks with dense vector search and metadata filters."""
        self._validate_configuration(chunking_strategy, embedding_model)

        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError(
                "Provide the tenant_id from the authenticated user's context."
            )
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")
        if n_results <= 0:
            raise ValueError("n_results must be positive.")

        collection = self.get_collection(chunking_strategy, embedding_model)
        if collection.count() == 0:
            return []

        # Centralized validation and construction of Chroma's `where` filter.
        where_clause = build_metadata_where(
            tenant_id=tenant_id,
            chunking_strategy=chunking_strategy,
            filters=filters,
        )

        query_embedding = self.embedding_manager.embed_query(
            question.strip(),
            embedding_model,
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict[str, Any]] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            output.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata or {},
                    "distance": float(distance),
                }
            )

        return output

    def get_chunks_by_ids(
        self,
        chunk_ids: Sequence[str],
        chunking_strategy: str,
        embedding_model: str,
        tenant_id: str,
    ) -> list[Document]:
        """Fetch chunks by ID while restricting results to the given tenant.

        This can be used to fetch parent chunks after hierarchical child search.
        """
        self._validate_configuration(chunking_strategy, embedding_model)

        if not chunk_ids:
            return []
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("A valid tenant_id is required.")

        collection = self.get_collection(chunking_strategy, embedding_model)
        results = collection.get(
            ids=list(chunk_ids),
            where={"tenant_id": tenant_id},
            include=["documents", "metadatas"],
        )

        documents: list[Document] = []
        for text, metadata in zip(
            results["documents"] or [],
            results["metadatas"] or [],
        ):
            if text is not None:
                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata or {},
                    )
                )

        return documents
