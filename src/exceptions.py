"""Custom exceptions and a helper that turns any error into a safe UI message.

Every ``RAGError`` carries a ``user_message`` that is safe to show to end users.
Technical details are logged instead of being displayed.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all application errors."""

    default_message = "Something went wrong. Please try again."

    def __init__(self, user_message: str | None = None) -> None:
        self.user_message = user_message or self.default_message
        super().__init__(self.user_message)


class ConfigurationError(RAGError):
    default_message = "The application is not configured correctly. Check your .env file."


class UnsupportedFileTypeError(RAGError):
    default_message = "This file type is not supported."


class FileValidationError(RAGError):
    default_message = "The uploaded file failed validation."


class CorruptedDocumentError(RAGError):
    default_message = "The document could not be read. It may be corrupted or password-protected."


class EmptyDocumentError(RAGError):
    default_message = "The document does not contain any readable text."


class EmbeddingError(RAGError):
    default_message = "Could not create embeddings for the document. Please try again."


class VectorStoreError(RAGError):
    default_message = "The vector database reported an error. Please try again."


class LLMError(RAGError):
    default_message = "The language model could not generate an answer. Please try again."


def _openai_message(exc: BaseException) -> str | None:
    """Return a friendly message if ``exc`` is a known OpenAI SDK error."""
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency
        return None

    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file."
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI rate limit or quota reached. Check your usage and billing, then retry."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not connect to OpenAI. Check your internet connection and retry."
    if isinstance(exc, openai.NotFoundError):
        return "The selected OpenAI model was not found or your account cannot use it."
    if isinstance(exc, openai.OpenAIError):
        return "OpenAI returned an error. Please try again in a moment."
    return None


def to_user_message(exc: BaseException) -> str:
    """Convert any exception into a message that is safe to display."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        specific = _openai_message(current)
        if specific:
            return specific
        current = current.__cause__ or current.__context__

    if isinstance(exc, RAGError):
        return exc.user_message
    return RAGError.default_message
