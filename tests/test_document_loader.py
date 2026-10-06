"""Tests for src/document_loader.py (PDF, DOCX, TXT, CSV loading)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.document_loader import load_document, supported_extensions
from src.exceptions import EmptyDocumentError, UnsupportedFileTypeError


def test_supported_extensions_includes_all_required_formats() -> None:
    extensions = supported_extensions()
    assert {".pdf", ".docx", ".txt", ".csv"}.issubset(set(extensions))


def test_load_pdf_extracts_text_per_page(sample_pdf: Path) -> None:
    loaded = load_document(sample_pdf, filename="sample.pdf", doc_id="doc1")
    assert loaded.page_count == 2
    assert len(loaded.sections) == 2
    assert "python" in loaded.sections[0].page_content.lower()
    assert loaded.sections[0].metadata["page"] == 1
    assert loaded.sections[0].metadata["source"] == "sample.pdf"
    assert loaded.sections[0].metadata["doc_id"] == "doc1"


def test_load_docx_extracts_paragraphs(sample_docx: Path) -> None:
    loaded = load_document(sample_docx, filename="sample.docx", doc_id="doc2")
    assert loaded.file_type == "docx"
    assert "supervised learning" in loaded.sections[0].page_content.lower()


def test_load_txt(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Plain text content for testing.", encoding="utf-8")
    loaded = load_document(path, filename="notes.txt", doc_id="doc3")
    assert loaded.sections[0].page_content == "Plain text content for testing."


def test_load_csv_groups_rows_into_sections(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")
    loaded = load_document(path, filename="data.csv", doc_id="doc4")
    assert loaded.file_type == "csv"
    assert "Alice" in loaded.sections[0].page_content
    assert "rows" in loaded.sections[0].metadata


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not really an image")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(path, filename="image.png", doc_id="doc5")


def test_empty_txt_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_document(path, filename="empty.txt", doc_id="doc6")
