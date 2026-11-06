from typing import Any

from vectorstore.chroma_manager import ChromaManager


def dense_search(
    question: str,
    chroma_manager: ChromaManager,
    chunking_strategy: str,
    embedding_model: str,
    tenant_id: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-k relevant chunks using dense vector search.

    Args:
        question: User's natural-language question.
        chroma_manager: Initialized ChromaManager instance.
        chunking_strategy: Selected chunking strategy.
        embedding_model: Embedding model used by the collection.
        tenant_id: Authorized tenant ID from the authenticated session.
        top_k: Maximum number of chunks to retrieve.
        filters: Optional additional metadata filters.

    Returns:
        A list of dictionaries containing chunk text,
        metadata, ID, and vector distance.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("A valid tenant_id is required.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # Use the existing ChromaManager to:
    # 1. Select the correct collection.
    # 2. Embed the question with the matching model.
    # 3. Apply tenant and metadata filters.
    # 4. Retrieve the nearest chunks.
    results = chroma_manager.search(
        question=question.strip(),
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
        tenant_id=tenant_id,
        n_results=top_k,
        filters=filters,
    )

    return results