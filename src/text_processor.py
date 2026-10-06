"""Text cleaning and chunking.

Chunking uses ``RecursiveCharacterTextSplitter``: it tries to split on paragraphs
first, then lines, then sentences/words, so chunks stay as meaningful as possible.
Every chunk inherits its page's metadata and gets a readable ``chunk_id``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.exceptions import EmptyDocumentError

logger = logging.getLogger(__name__)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_INLINE_SPACES = re.compile(r"[ \t\f\v]+")
_SPACE_AROUND_NEWLINE = re.compile(r" ?\n ?")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize extracted text without changing its meaning.

    * NFKC normalization (turns ligatures like 'ﬁ' into 'fi')
    * removes control characters
    * re-joins words hyphenated across line breaks
    * collapses repeated spaces and blank lines
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _INLINE_SPACES.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _chunk_label(metadata: Mapping[str, Any], index: int) -> str:
    """Readable chunk id such as 'example_5_03' (page 5, third chunk)."""
    stem = Path(str(metadata.get("source", "document"))).stem
    if metadata.get("page") is not None:
        return f"{stem}_{metadata['page']}_{index:02d}"
    if metadata.get("rows"):
        return f"{stem}_rows{metadata['rows']}_{index:02d}"
    return f"{stem}_{index:02d}"


def split_documents(
    sections: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Clean each section and split it into overlapping chunks with metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks: list[Document] = []
    for section in sections:
        cleaned = clean_text(section.page_content)
        if not cleaned:
            continue
        cleaned_section = Document(page_content=cleaned, metadata=dict(section.metadata))
        for index, chunk in enumerate(splitter.split_documents([cleaned_section]), start=1):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = _chunk_label(chunk.metadata, index)
            chunks.append(chunk)

    if not chunks:
        raise EmptyDocumentError()
    logger.info("Chunks created: %d (size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)
    return chunks
