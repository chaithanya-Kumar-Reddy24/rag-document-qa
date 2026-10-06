"""Small, dependency-free helpers: logging, filenames, hashing, formatting."""

from __future__ import annotations

import hashlib
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from src.exceptions import (
    EmptyDocumentError,
    FileValidationError,
    UnsupportedFileTypeError,
)

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_HANDLER_TAG = "_rag_qa_handler"


class RedactingFilter(logging.Filter):
    """Masks anything that looks like an OpenAI API key before it is written."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _API_KEY_PATTERN.sub("sk-***REDACTED***", message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure console + rotating file logging (safe to call repeatedly)."""
    root = logging.getLogger()
    if any(getattr(handler, _HANDLER_TAG, False) for handler in root.handlers):
        return

    root.setLevel(level)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(
                RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            )
        except OSError:
            pass  # file logging is optional

    formatter = logging.Formatter(_LOG_FORMAT)
    redactor = RedactingFilter()
    for handler in handlers:
        setattr(handler, _HANDLER_TAG, True)
        handler.setFormatter(formatter)
        handler.addFilter(redactor)
        root.addHandler(handler)

    for noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def sanitize_filename(name: str) -> str:
    """Return a safe filename: no directories, no odd characters, lowercase extension."""
    base = PurePosixPath(str(name).replace("\\", "/")).name
    suffix = _UNSAFE_FILENAME_CHARS.sub("", Path(base).suffix.lower())
    stem = _UNSAFE_FILENAME_CHARS.sub("_", Path(base).stem).strip("._-")[:80]
    return f"{stem or 'document'}{suffix}"


def validate_upload(
    filename: str,
    size_bytes: int,
    allowed_extensions: Iterable[str],
    max_mb: int,
) -> str:
    """Validate an upload and return its sanitized filename."""
    safe_name = sanitize_filename(filename)
    extension = Path(safe_name).suffix.lower()
    allowed = tuple(allowed_extensions)

    if extension not in allowed:
        raise UnsupportedFileTypeError(
            f"'{safe_name}' is not supported. Allowed types: {', '.join(allowed)}."
        )
    if size_bytes <= 0:
        raise EmptyDocumentError(f"'{safe_name}' is empty.")
    if size_bytes > max_mb * 1024 * 1024:
        raise FileValidationError(f"'{safe_name}' is larger than the {max_mb} MB limit.")
    return safe_name


def compute_hash(data: bytes) -> str:
    """Short content hash used as the document ID (also for duplicate detection)."""
    return hashlib.sha256(data).hexdigest()[:16]


def describe_location(metadata: Mapping[str, Any]) -> str:
    """Human-readable position of a chunk: 'Page 4', 'Rows 1-50' or ''."""
    if metadata.get("page") is not None:
        return f"Page {metadata['page']}"
    if metadata.get("rows"):
        return f"Rows {metadata['rows']}"
    return ""


def truncate(text: str, limit: int = 200) -> str:
    """Shorten text for safe logging."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def chunk_uid(metadata: Mapping[str, Any]) -> str:
    """Stable chunk identifier, also used as the Chroma document id.

    Combining ``doc_id`` (content hash of the file) and ``chunk_id`` (position
    within that file) is what lets vector search and keyword search results be
    matched up as "the same chunk" during hybrid fusion.
    """
    return f"{metadata['doc_id']}:{metadata['chunk_id']}"
