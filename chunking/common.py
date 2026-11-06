from collections.abc import Mapping
from typing import Any


def document_unit_id(metadata: Mapping[str, Any]) -> str:
    """Return a stable ID for a document text unit, including PDF pages."""
    document_id = str(
        metadata.get("document_id")
        or metadata.get("source", "document")
    )
    page = metadata.get("page")
    if page is not None:
        return f"{document_id}:page:{page}"
    return document_id
