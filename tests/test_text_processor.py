"""Tests for src/text_processor.py (cleaning + chunking + metadata)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.exceptions import EmptyDocumentError
from src.text_processor import clean_text, split_documents


def test_clean_text_rejoins_hyphenated_linebreaks() -> None:
    assert clean_text("infor-\nmation") == "information"


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("too    many   spaces") == "too many spaces"


def test_clean_text_collapses_blank_lines() -> None:
    result = clean_text("line one\n\n\n\n\nline two")
    assert result == "line one\n\nline two"


def test_split_documents_preserves_metadata_and_adds_chunk_id() -> None:
    text = "Sentence one. " * 40
    section = Document(page_content=text, metadata={"source": "example.pdf", "page": 5})
    chunks = split_documents([section], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "example.pdf"
        assert chunk.metadata["page"] == 5
        assert chunk.metadata["chunk_id"].startswith("example_5_")


def test_split_documents_respects_chunk_size_overlap_validation() -> None:
    section = Document(page_content="text", metadata={})
    with pytest.raises(ValueError):
        split_documents([section], chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        split_documents([section], chunk_size=0, chunk_overlap=0)


def test_split_documents_raises_when_nothing_left_after_cleaning() -> None:
    section = Document(page_content="   \n\n  ", metadata={})
    with pytest.raises(EmptyDocumentError):
        split_documents([section])
