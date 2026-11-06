import re
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.common import document_unit_id


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _extract_sections(
    text: str,
    default_title: str = "Document body",
) -> list[dict]:
    """Divide Markdown-like text at heading boundaries."""

    sections = []

    current_title = default_title
    current_level = 0
    current_lines = []

    for line in text.splitlines():
        match = HEADING_PATTERN.match(line.strip())

        if match:
            body = "\n".join(current_lines).strip()

            if body:
                sections.append({
                    "title": current_title,
                    "heading_level": current_level,
                    "text": body,
                })

            current_title = match.group(2).strip()
            current_level = len(match.group(1))
            current_lines = []

        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()

    if body:
        sections.append({
            "title": current_title,
            "heading_level": current_level,
            "text": body,
        })

    # Preserve a document that contains only headings or no body text.
    if not sections and text.strip():
        sections.append({
            "title": current_title,
            "heading_level": current_level,
            "text": text.strip(),
        })

    return sections


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Split documents by heading sections, then chunk within each section.

    Markdown headings such as '# Billing' and '## Complaints' are
    treated as structural boundaries.
    """

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
        strip_whitespace=True,
    )

    chunks = []

    for document in documents:
        sections = _extract_sections(
            document.page_content,
            default_title=str(
                document.metadata.get("section", "Document body")
            ),
        )
        unit_id = document_unit_id(document.metadata)
        document_chunk_index = 0

        for section_index, section in enumerate(sections):
            section_text = section["text"]

            section_parts = splitter.split_text(section_text)

            for section_part in section_parts:
                # Repeat the heading in each chunk for context.
                heading_level = section["heading_level"]
                title = section["title"]

                if heading_level > 0:
                    chunk_text = (
                        f"{'#' * heading_level} {title}\n\n"
                        f"{section_part}"
                    )
                else:
                    chunk_text = section_part

                metadata = dict(document.metadata)
                chunk_index = document_chunk_index
                document_chunk_index += 1

                metadata.update({
                    "chunking_strategy": "structure_aware",
                    "chunk_index": chunk_index,
                    "chunk_id": (
                        f"{unit_id}:structure:{chunk_index}"
                    ),
                    "section_title": title,
                    "section_index": section_index,
                    "heading_level": heading_level,
                    "chunk_size_chars": len(chunk_text),
                })

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=metadata,
                    )
                )

    return chunks