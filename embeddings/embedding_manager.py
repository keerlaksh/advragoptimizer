from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


# All embedding models used in RAGOptimizer.
EMBEDDING_MODELS = {
    "minilm": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "document_prefix": "",
    },
    "e5-small": {
        "model_id": "intfloat/e5-small",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "bge-base": {
        "model_id": "BAAI/bge-base-en-v1.5",
        "query_prefix": (
            "Represent this sentence for searching relevant passages: "
        ),
        "document_prefix": "",
    },
}


@lru_cache(maxsize=3)
def _load_model(model_id: str) -> SentenceTransformer:
    """Load each embedding model once and reuse it."""
    return SentenceTransformer(model_id)


class EmbeddingManager:
    """Generate document and query embeddings for RAGOptimizer."""

    def __init__(self):
        self.models = EMBEDDING_MODELS

    def get_model(self, model_name: str) -> SentenceTransformer:
        """Load a model using its short name."""
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(self.models)}"
            )

        model_id = self.models[model_name]["model_id"]
        return _load_model(model_id)

    def _encode(
        self,
        texts: list[str],
        model_name: str,
        prefix_type: str,
    ) -> list[list[float]]:
        """Encode and normalize a list of texts."""

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip()
               for text in texts):
            raise ValueError("All input texts must be non-empty strings.")

        if prefix_type not in ("query", "document"):
            raise ValueError("prefix_type must be 'query' or 'document'.")

        if model_name not in self.models:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(self.models)}"
            )

        model_config = self.models[model_name]
        prefix = model_config[f"{prefix_type}_prefix"]

        prepared_texts = [
            prefix + text for text in texts
        ]

        model = self.get_model(model_name)

        vectors = model.encode(
            prepared_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_documents(
        self,
        texts: list[str],
        model_name: str,
    ) -> list[list[float]]:
        """Generate embeddings for document chunks."""

        return self._encode(
            texts=texts,
            model_name=model_name,
            prefix_type="document",
        )

    def embed_query(
        self,
        query: str,
        model_name: str,
    ) -> list[float]:
        """Generate an embedding for a user's question."""

        vectors = self._encode(
            texts=[query],
            model_name=model_name,
            prefix_type="query",
        )

        return vectors[0]

    def get_embedding_dimension(self, model_name: str) -> int:
        """Return the embedding dimension of a model."""

        model = self.get_model(model_name)
        dimension = model.get_sentence_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                f"Could not determine the dimension for {model_name}."
            )

        return int(dimension)