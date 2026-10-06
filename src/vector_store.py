"""ChromaDB vector store wrapper.

Responsibilities: persistent storage, adding chunks, duplicate lookup, metadata
filtering (search only selected documents), listing/deleting documents and
clearing the collection.

The collection uses cosine distance (``hnsw:space = cosine``), so
``relevance = 1 - distance`` is the cosine similarity between question and chunk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.exceptions import EmbeddingError, VectorStoreError
from src.utils import chunk_uid

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64


@dataclass(frozen=True)
class IndexedDocument:
    """Summary of one document stored in the vector database."""

    doc_id: str
    filename: str
    file_type: str
    pages: int
    chunks: int


def _is_embedding_failure(exc: BaseException) -> bool:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, openai.OpenAIError)


class VectorStoreManager:
    """Thin, testable layer over ``langchain_chroma.Chroma``."""

    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._store = self._create_store()
        self.version = 0  # bumped on every write; lets callers cache derived indexes (e.g. BM25)

    def _create_store(self) -> Chroma:
        try:
            self._settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
            return Chroma(
                collection_name=self._settings.collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(self._settings.chroma_persist_directory),
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception("Could not open ChromaDB")
            raise VectorStoreError("Could not open the local vector database.") from exc

    # ------------------------------------------------------------------ write
    def has_document(self, doc_id: str) -> bool:
        """True if a document with this content hash is already indexed."""
        try:
            return bool(self._store.get(where={"doc_id": doc_id}, limit=1, include=[])["ids"])
        except Exception as exc:
            logger.exception("Duplicate check failed")
            raise VectorStoreError() from exc

    def add_chunks(self, chunks: Sequence[Document]) -> int:
        """Embed and store chunks. Rolls back the document if anything fails."""
        if not chunks:
            return 0
        doc_id = str(chunks[0].metadata["doc_id"])
        try:
            for start in range(0, len(chunks), _BATCH_SIZE):
                batch = list(chunks[start : start + _BATCH_SIZE])
                ids = [chunk_uid(c.metadata) for c in batch]
                self._store.add_documents(batch, ids=ids)
        except Exception as exc:
            logger.exception("Vector indexing failed for doc_id=%s", doc_id)
            self._rollback(doc_id)
            if _is_embedding_failure(exc):
                raise EmbeddingError() from exc
            raise VectorStoreError("Could not store the document in the vector database.") from exc
        self.version += 1
        logger.info("Vector indexing complete: doc_id=%s, %d chunks", doc_id, len(chunks))
        return len(chunks)

    def _rollback(self, doc_id: str) -> None:
        try:
            self.delete_document(doc_id)
        except Exception:  # best effort only
            logger.warning("Rollback failed for doc_id=%s", doc_id)

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk that belongs to one document."""
        try:
            ids = self._store.get(where={"doc_id": doc_id}, include=[])["ids"]
            if ids:
                self._store.delete(ids=ids)
        except Exception as exc:
            logger.exception("Could not delete doc_id=%s", doc_id)
            raise VectorStoreError() from exc
        self.version += 1

    def clear(self) -> None:
        """Delete the whole collection and start with a fresh, empty one."""
        try:
            self._store.delete_collection()
        except Exception as exc:
            logger.exception("Could not clear the collection")
            raise VectorStoreError("Could not clear the vector database.") from exc
        self._store = self._create_store()
        self.version += 1
        logger.info("Vector database cleared")

    # ------------------------------------------------------------------- read
    def count_chunks(self) -> int:
        try:
            return len(self._store.get(include=[])["ids"])
        except Exception as exc:
            raise VectorStoreError() from exc

    def get_all_chunks(self) -> list[tuple[str, Document]]:
        """Every stored chunk as ``(id, Document)`` pairs, for building a keyword index."""
        try:
            data = self._store.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.exception("Could not fetch chunks for keyword indexing")
            raise VectorStoreError() from exc
        ids = data.get("ids") or []
        texts = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        return [
            (uid, Document(page_content=text or "", metadata=dict(meta or {})))
            for uid, text, meta in zip(ids, texts, metadatas)
        ]

    def list_documents(self) -> list[IndexedDocument]:
        """Aggregate chunk metadata into one row per document."""
        try:
            metadatas: list[dict[str, Any]] = self._store.get(include=["metadatas"])["metadatas"] or []
        except Exception as exc:
            logger.exception("Could not list documents")
            raise VectorStoreError() from exc

        grouped: dict[str, dict[str, Any]] = {}
        for meta in metadatas:
            entry = grouped.setdefault(
                str(meta.get("doc_id")),
                {
                    "filename": meta.get("source", "unknown"),
                    "file_type": meta.get("file_type", ""),
                    "pages": int(meta.get("page_count", 0) or 0),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        return sorted(
            (IndexedDocument(doc_id=key, **value) for key, value in grouped.items()),
            key=lambda doc: doc.filename.lower(),
        )

    def search(
        self,
        query: str,
        *,
        k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> list[tuple[Document, float]]:
        """Semantic search returning ``(chunk, cosine_similarity)`` pairs.

        ``doc_ids=None`` searches everything; a list restricts the search to those
        documents through a Chroma metadata filter; an empty list matches nothing.
        """
        if doc_ids is not None and len(doc_ids) == 0:
            return []
        if self.count_chunks() == 0:
            return []

        where: dict[str, Any] | None = None
        if doc_ids is not None:
            ids = list(doc_ids)
            where = {"doc_id": ids[0]} if len(ids) == 1 else {"doc_id": {"$in": ids}}

        try:
            results = self._store.similarity_search_with_score(query, k=k, filter=where)
        except Exception as exc:
            logger.exception("Similarity search failed")
            if _is_embedding_failure(exc):
                raise EmbeddingError("Could not embed your question. Please try again.") from exc
            raise VectorStoreError("The similarity search failed. Please try again.") from exc
        return [(doc, 1.0 - float(distance)) for doc, distance in results]
