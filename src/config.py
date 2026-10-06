"""Application configuration.

All settings come from environment variables loaded from ``.env``.
This version uses local Ollama for the chat model and local
Sentence Transformers for embeddings. No OpenAI API key is required.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Keep ChromaDB quiet.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logger = logging.getLogger(__name__)

OLLAMA_MODEL_DEFAULT = "llama3"
EMBEDDING_MODEL_DEFAULT = "all-MiniLM-L6-v2"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default

    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer for %s; using default %s",
            name,
            default,
        )
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default

    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid number for %s; using default %s",
            name,
            default,
        )
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()

    if not raw:
        return default

    if raw in {"1", "true", "yes", "on"}:
        return True

    if raw in {"0", "false", "no", "off"}:
        return False

    logger.warning(
        "Invalid boolean for %s; using default %s",
        name,
        default,
    )
    return default


def _env_path(name: str, default: str) -> Path:
    """Read a path; relative paths are resolved against the project root."""

    path = Path(os.getenv(name, "").strip() or default)

    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings for the local RAG system."""

    # Backward compatibility only.
    # Existing tests may still pass this argument, but the local
    # application does NOT use OpenAI.
    openai_api_key: str = ""

    # Local Ollama chat model.
    chat_model: str = OLLAMA_MODEL_DEFAULT

    # Local Sentence Transformers embedding model.
    embedding_model: str = EMBEDDING_MODEL_DEFAULT

    # ChromaDB configuration.
    chroma_persist_directory: Path = (
        PROJECT_ROOT / "data" / "chroma_db"
    )

    # Uploaded document directory.
    upload_directory: Path = PROJECT_ROOT / "data" / "uploads"

    # Application log file.
    log_file: Path = PROJECT_ROOT / "data" / "logs" / "app.log"

    # Chroma collection.
    collection_name: str = "documents"

    # Text chunking.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval.
    top_k: int = 4
    relevance_threshold: float = 0.3

    # Ollama generation.
    temperature: float = 0.0

    # Upload validation.
    max_upload_mb: int = 200

    # Conversation memory.
    max_history_turns: int = 6

    # Maximum context passed to the LLM.
    max_context_chars: int = 12000

    # Hybrid retrieval.
    enable_hybrid_search: bool = True
    candidate_pool_size: int = 10
    rrf_k: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""

        return cls(
            chat_model=(
                os.getenv("OLLAMA_MODEL", "").strip()
                or OLLAMA_MODEL_DEFAULT
            ),
            embedding_model=(
                os.getenv("EMBEDDING_MODEL", "").strip()
                or EMBEDDING_MODEL_DEFAULT
            ),
            chroma_persist_directory=_env_path(
                "CHROMA_PERSIST_DIRECTORY",
                "./data/chroma_db",
            ),
            upload_directory=_env_path(
                "UPLOAD_DIRECTORY",
                "./data/uploads",
            ),
            chunk_size=_env_int(
                "CHUNK_SIZE",
                1000,
            ),
            chunk_overlap=_env_int(
                "CHUNK_OVERLAP",
                200,
            ),
            top_k=_env_int(
                "TOP_K",
                4,
            ),
            relevance_threshold=_env_float(
                "RELEVANCE_THRESHOLD",
                0.3,
            ),
            max_upload_mb=_env_int(
                "MAX_UPLOAD_MB",
                200,
            ),
            enable_hybrid_search=_env_bool(
                "HYBRID_SEARCH",
                True,
            ),
            candidate_pool_size=_env_int(
                "CANDIDATE_POOL_SIZE",
                10,
            ),
            rrf_k=_env_int(
                "RRF_K",
                60,
            ),
        )

    @property
    def ollama_model(self) -> str:
        """Return the configured Ollama model name."""

        return self.chat_model

    def validate(self) -> None:
        """Validate local-model configuration."""

        if not self.chat_model:
            raise ConfigurationError(
                "OLLAMA_MODEL is missing. "
                "Set OLLAMA_MODEL=llama3 in your .env file."
            )

        if not self.embedding_model:
            raise ConfigurationError(
                "EMBEDDING_MODEL is missing. "
                "Set EMBEDDING_MODEL=all-MiniLM-L6-v2 in your .env file."
            )