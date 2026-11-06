from typing import Any


# Metadata fields users may filter by.
# These fields must exist on the stored chunks to work.
ALLOWED_FILTER_FIELDS = {
    "source",
    "file_type",
    "document_id",
    "page",
    "section",
    "section_title",
    "section_index",
    "heading_level",
    "document_type",
    "effective_date",
    "config_id",
    "chunking_strategy",
    "embedding_model",
}

# These fields are controlled by the backend, not the user.
PROTECTED_FIELDS = {
    "tenant_id",
    "chunk_level",
}

VALID_CHUNKING_STRATEGIES = {
    "fixed_size",
    "semantic",
    "hierarchical",
    "structure_aware",
}


def validate_metadata_filters(
    filters: dict[str, Any] | None = None,
) -> dict[str, str | int | float | bool]:
    """
    Validate optional metadata filters.

    This initial implementation supports exact-match filters only.
    """

    if filters is None:
        return {}

    if not isinstance(filters, dict):
        raise TypeError("Metadata filters must be a dictionary.")

    validated = {}

    for field, value in filters.items():
        if field in PROTECTED_FIELDS:
            raise ValueError(
                f"'{field}' is controlled by the backend "
                "and cannot be supplied as an optional filter."
            )

        if field not in ALLOWED_FILTER_FIELDS:
            raise ValueError(
                f"Unsupported metadata field: '{field}'. "
                f"Allowed fields: {sorted(ALLOWED_FILTER_FIELDS)}"
            )

        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(
                f"Filter '{field}' must use a string, integer, "
                "float, or boolean value."
            )

        validated[field] = value

    return validated


def build_metadata_where(
    tenant_id: str,
    chunking_strategy: str,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a ChromaDB 'where' filter.

    Every search must be restricted to the authenticated tenant.
    Hierarchical retrieval searches child chunks, not parent chunks.
    """

    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError(
            "A valid tenant_id from the authenticated user is required."
        )

    if chunking_strategy not in VALID_CHUNKING_STRATEGIES:
        raise ValueError(
            f"Unsupported chunking strategy: {chunking_strategy}"
        )

    validated_filters = validate_metadata_filters(filters)

    # Always start with the tenant restriction.
    conditions: list[dict[str, Any]] = [
        {"tenant_id": tenant_id}
    ]

    # Search child chunks for hierarchical retrieval.
    if chunking_strategy == "hierarchical":
        conditions.append({"chunk_level": "child"})

    # Add optional document/source filters.
    for field, value in validated_filters.items():
        conditions.append({field: value})

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}