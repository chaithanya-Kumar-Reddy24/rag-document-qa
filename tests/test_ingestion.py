"""Tests for src/ingestion.py (upload validation, duplicate detection, rollback)."""

from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.ingestion import IngestionService
from src.vector_store import VectorStoreManager


def _service(tmp_settings: Settings, vector_store: VectorStoreManager) -> IngestionService:
    return IngestionService(tmp_settings, vector_store)


def test_ingest_indexes_a_valid_txt_file(tmp_settings: Settings, vector_store: VectorStoreManager) -> None:
    service = _service(tmp_settings, vector_store)
    result = service.ingest("notes.txt", b"Python is used for machine learning and data science.")
    assert result.status == "indexed"
    assert result.chunks > 0
    assert vector_store.count_chunks() == result.chunks


def test_ingest_detects_duplicate_content(tmp_settings: Settings, vector_store: VectorStoreManager) -> None:
    service = _service(tmp_settings, vector_store)
    data = b"The same content uploaded twice."
    first = service.ingest("a.txt", data)
    second = service.ingest("a_copy.txt", data)
    assert first.status == "indexed"
    assert second.status == "duplicate"
    assert vector_store.count_chunks() == first.chunks  # not doubled


def test_ingest_rejects_unsupported_extension(tmp_settings: Settings, vector_store: VectorStoreManager) -> None:
    service = _service(tmp_settings, vector_store)
    result = service.ingest("image.png", b"not a real image")
    assert result.status == "failed"
    assert "not supported" in result.message.lower()


def test_ingest_rejects_empty_file(tmp_settings: Settings, vector_store: VectorStoreManager) -> None:
    service = _service(tmp_settings, vector_store)
    result = service.ingest("empty.txt", b"")
    assert result.status == "failed"


def test_ingest_rejects_oversized_file(tmp_settings: Settings, vector_store: VectorStoreManager) -> None:
    tmp_settings = Settings(
        openai_api_key="sk-test",
        chroma_persist_directory=tmp_settings.chroma_persist_directory,
        upload_directory=tmp_settings.upload_directory,
        max_upload_mb=1,
    )
    service = _service(tmp_settings, vector_store)
    result = service.ingest("big.txt", b"x" * (2 * 1024 * 1024))
    assert result.status == "failed"
    assert "mb" in result.message.lower()


def test_ingest_does_not_leave_orphan_file_on_failure(
    tmp_settings: Settings, vector_store: VectorStoreManager
) -> None:
    service = _service(tmp_settings, vector_store)
    service.ingest("empty.txt", b"")
    leftovers = list(tmp_settings.upload_directory.glob("*")) if tmp_settings.upload_directory.exists() else []
    assert leftovers == []
