from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from typing import Any

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _base_metadata(
    filename: str,
    file_type: str,
    document_id: str,
    tenant_id: str | None,
) -> dict[str, str | int]:
    """Create metadata shared by all text units from one file."""

    metadata: dict[str, str | int] = {
        "source": filename,
        "file_type": file_type,
        "document_id": document_id,
    }

    if tenant_id is not None:
        tenant_id = tenant_id.strip()

        if not tenant_id:
            raise ValueError("tenant_id cannot be empty.")

        metadata["tenant_id"] = tenant_id

    return metadata


def _extract_docx_text(data: bytes) -> str:
    """Extract Word paragraphs, headings, and tables in document order."""

    word_document = DocxDocument(BytesIO(data))
    blocks: list[str] = []

    for element in word_document.element.body.iterchildren():

        # Extract paragraphs and preserve heading levels.
        if element.tag == qn("w:p"):
            paragraph = Paragraph(element, word_document)
            text = paragraph.text.strip()

            if not text:
                continue

            style_name = (
                paragraph.style.name
                if paragraph.style is not None
                else ""
            )

            if style_name.startswith("Heading"):
                try:
                    level = int(style_name.split()[-1])
                except ValueError:
                    level = 1

                level = max(1, min(level, 6))
                blocks.append(f"{'#' * level} {text}")

            elif style_name == "Title":
                blocks.append(f"# {text}")

            else:
                blocks.append(text)

        # Convert tables into readable text.
        elif element.tag == qn("w:tbl"):
            table = Table(element, word_document)
            rows = [
                [
                    cell.text.replace("\n", " ").strip()
                    for cell in row.cells
                ]
                for row in table.rows
            ]

            if not rows:
                continue

            table_lines = [
                " | ".join(row)
                for row in rows
            ]

            # Add a Markdown-style separator after the first row.
            if rows[0]:
                separator = " | ".join(
                    "---" for _ in rows[0]
                )
                table_lines.insert(1, separator)

            blocks.append("\n".join(table_lines))

    return "\n\n".join(blocks).strip()


def _load_from_bytes(
    data: bytes,
    filename: str,
    tenant_id: str | None = None,
) -> list[Document]:
    """Extract a file into LangChain Documents with metadata."""

    if not data:
        raise ValueError(f"The uploaded file '{filename}' is empty.")

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    content_hash = hashlib.sha256(data).hexdigest()
    document_identity = "\0".join(
        (
            (tenant_id or "").strip(),
            Path(filename).name,
            content_hash,
        )
    )
    document_id = uuid5(NAMESPACE_URL, document_identity).hex

    base_metadata = _base_metadata(
        filename=Path(filename).name,
        file_type=extension.removeprefix("."),
        document_id=document_id,
        tenant_id=tenant_id,
    )

    documents: list[Document] = []

    # PDF: keep each page as a separate loaded document.
    if extension == ".pdf":
        try:
            pdf = PdfReader(BytesIO(data))

            for page_number, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()

                if not text:
                    continue

                metadata = {
                    **base_metadata,
                    "page": page_number,
                }

                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

        except Exception as exc:
            raise ValueError(
                f"Could not read PDF '{filename}': {exc}"
            ) from exc

        if not documents:
            raise ValueError(
                f"No extractable text was found in '{filename}'. "
                "If this is a scanned PDF, OCR may be required."
            )

    # DOCX: preserve headings, paragraphs, and tables.
    elif extension == ".docx":
        try:
            text = _extract_docx_text(data)
        except Exception as exc:
            raise ValueError(
                f"Could not read DOCX '{filename}': {exc}"
            ) from exc

        if not text:
            raise ValueError(
                f"No extractable text was found in '{filename}'."
            )

        documents.append(
            Document(
                page_content=text,
                metadata=base_metadata.copy(),
            )
        )

    # TXT and Markdown: read the original text.
    else:
        text = data.decode(
            "utf-8-sig",
            errors="replace",
        ).strip()

        if not text:
            raise ValueError(
                f"No readable text was found in '{filename}'."
            )

        documents.append(
            Document(
                page_content=text,
                metadata=base_metadata.copy(),
            )
        )

    return documents


def load_document(
    file_path: str | Path,
    tenant_id: str | None = None,
) -> list[Document]:
    """Load a document from a local file path."""

    path = Path(file_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return _load_from_bytes(
        data=path.read_bytes(),
        filename=path.name,
        tenant_id=tenant_id,
    )


def load_uploaded_file(
    uploaded_file: Any,
    tenant_id: str | None = None,
) -> list[Document]:
    """Load a file received from Streamlit's file uploader."""

    filename = getattr(uploaded_file, "name", None)

    if not filename:
        raise ValueError(
            "The uploaded file must have a filename."
        )

    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()

    if not isinstance(data, bytes):
        raise TypeError("The uploaded file must provide bytes.")

    return _load_from_bytes(
        data=data,
        filename=filename,
        tenant_id=tenant_id,
    )