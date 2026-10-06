"""Ingestion service: upload -> validate -> load -> clean -> chunk -> embed -> store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config import Settings
from src.document_loader import load_document, supported_extensions
from src.exceptions import RAGError, to_user_message
from src.text_processor import split_documents
from src.utils import compute_hash, validate_upload
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Outcome of indexing one uploaded file."""

    filename: str
    status: Literal["indexed", "duplicate", "failed"]
    message: str
    doc_id: str = ""
    file_type: str = ""
    pages: int = 0
    chunks: int = 0


class IngestionService:
    """Turns raw uploaded bytes into indexed, searchable chunks."""

    def __init__(self, settings: Settings, vector_store: VectorStoreManager) -> None:
        self._settings = settings
        self._vector_store = vector_store

    def ingest(
        self,
        filename: str,
        data: bytes,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionResult:
        """Index one file. Never raises: failures are returned as ``failed`` results."""
        size = chunk_size or self._settings.chunk_size
        overlap = self._settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        logger.info("Document uploaded: %r (%d bytes)", filename, len(data))
        saved_path: Path | None = None
        try:
            safe_name = validate_upload(
                filename, len(data), supported_extensions(), self._settings.max_upload_mb
            )
            doc_id = compute_hash(data)

            if self._vector_store.has_document(doc_id):
                logger.info("Duplicate skipped: %s (doc_id=%s)", safe_name, doc_id)
                return IngestionResult(
                    filename=safe_name,
                    status="duplicate",
                    message="Already indexed - skipped.",
                    doc_id=doc_id,
                )

            saved_path = self._save_upload(doc_id, safe_name, data)
            loaded = load_document(saved_path, filename=safe_name, doc_id=doc_id)
            chunks = split_documents(loaded.sections, size, overlap)
            for chunk in chunks:
                chunk.metadata["page_count"] = loaded.page_count
            self._vector_store.add_chunks(chunks)

            return IngestionResult(
                filename=safe_name,
                status="indexed",
                message="Indexed successfully.",
                doc_id=doc_id,
                file_type=loaded.file_type,
                pages=loaded.page_count,
                chunks=len(chunks),
            )
        except RAGError as exc:
            self._discard(saved_path)
            logger.warning("Ingestion failed for %r: %s", filename, type(exc).__name__)
            return IngestionResult(filename=filename, status="failed", message=to_user_message(exc))
        except Exception as exc:
            self._discard(saved_path)
            logger.exception("Unexpected ingestion error for %r", filename)
            return IngestionResult(filename=filename, status="failed", message=to_user_message(exc))

    def _save_upload(self, doc_id: str, safe_name: str, data: bytes) -> Path:
        """Store the raw bytes under a sanitized name. Files are read, never executed."""
        directory = self._settings.upload_directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{doc_id}_{safe_name}"
        path.write_bytes(data)
        return path

    @staticmethod
    def _discard(path: Path | None) -> None:
        if path is not None:
            path.unlink(missing_ok=True)

    def remove_upload(self, doc_id: str) -> None:
        """Delete the stored copy of a document (its vectors are removed separately)."""
        for path in self._settings.upload_directory.glob(f"{doc_id}_*"):
            path.unlink(missing_ok=True)

    def clear_uploads(self) -> None:
        """Delete all stored uploads (used together with clearing the database)."""
        directory = self._settings.upload_directory
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file() and path.name != ".gitkeep":
                    path.unlink(missing_ok=True)
