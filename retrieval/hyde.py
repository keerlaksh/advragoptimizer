import os
from typing import Any

from generation.groq_client import create_chat_completion
from vectorstore.chroma_manager import ChromaManager
from retrieval.metadata_filters import build_metadata_where


def generate_hypothetical_document(
    question: str,
    model_name: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Use an LLM to generate a hypothetical passage for a question.

    This passage is a search aid, NOT a trusted answer.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Configure it in your environment."
        )

    model_name = model_name or os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    response = create_chat_completion(
        api_key,
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate a short hypothetical passage that could "
                    "appear in a company knowledge document and would "
                    "plausibly answer the user's question. "
                    "Use relevant terminology and explanatory wording. "
                    "Return only the passage. Do not discuss the task "
                    "or claim that the passage comes from a real document. "
                    "The passage is only for search and may contain "
                    "unverified information."
                ),
            },
            {
                "role": "user",
                "content": question.strip(),
            },
        ],
        temperature=0.2,
        max_completion_tokens=200,
    )

    hypothetical = response.choices[0].message.content

    if not hypothetical or not hypothetical.strip():
        raise RuntimeError(
            "The LLM returned an empty hypothetical passage."
        )

    return hypothetical.strip()


def _build_where_filter(
    tenant_id: str,
    chunking_strategy: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the metadata filter for the vector search.

    Every search is restricted to the authorized tenant.
    Hierarchical retrieval searches child chunks only.
    """

    return build_metadata_where(
        tenant_id=tenant_id,
        chunking_strategy=chunking_strategy,
        filters=filters,
    )


def hyde_search(
    question: str,
    chroma_manager: ChromaManager,
    chunking_strategy: str,
    embedding_model: str,
    tenant_id: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    hypothetical_document: str | None = None,
    llm_model_name: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve actual document chunks using HyDE.

    Steps:
        1. Generate a hypothetical passage, unless supplied.
        2. Embed that passage with the selected embedding model.
        3. Search the matching ChromaDB collection.
        4. Apply tenant and additional metadata filters.
        5. Return the top-k actual document chunks.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("A valid tenant_id is required.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    # Validate the requested configuration and get its collection.
    collection = chroma_manager.get_collection(
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
    )

    if collection.count() == 0:
        return []

    # Step 1: Generate the hypothetical passage.
    # Supplying hypothetical_document allows testing without an LLM call.
    if hypothetical_document is None:
        hypothetical_document = generate_hypothetical_document(
            question=question,
            model_name=llm_model_name,
            api_key=api_key,
        )

    if (
        not isinstance(hypothetical_document, str)
        or not hypothetical_document.strip()
    ):
        raise ValueError(
            "hypothetical_document must be a non-empty string."
        )

    # Step 2: Embed the hypothetical passage.
    #
    # Use embed_documents rather than embed_query because the generated
    # text is document-like. This also applies E5's "passage: " prefix.
    embedding_manager = chroma_manager.embedding_manager

    hypothetical_embedding = embedding_manager.embed_documents(
        [hypothetical_document.strip()],
        embedding_model,
    )[0]

    # Step 3: Build the authorized search filter.
    where_filter = _build_where_filter(
        tenant_id=tenant_id,
        chunking_strategy=chunking_strategy,
        filters=filters,
    )

    # Step 4: Search the real ChromaDB collection.
    # The hypothetical passage itself is NOT stored or returned as evidence.
    results = collection.query(
        query_embeddings=[hypothetical_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # Step 5: Return actual retrieved chunks.
    retrieved_chunks = []

    ids = results["ids"][0]
    texts = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, text, metadata, distance in zip(
        ids,
        texts,
        metadatas,
        distances,
    ):
        retrieved_chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata or {},
            "distance": float(distance),
            "retrieval_method": "hyde",
        })

    return retrieved_chunks