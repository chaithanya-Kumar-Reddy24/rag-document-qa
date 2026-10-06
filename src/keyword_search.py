"""Keyword (BM25) search — the other half of hybrid retrieval.

Vector search finds chunks that mean the same thing as the query, even with
different wording. It's weaker on exact terms it wasn't trained to distinguish
well: product codes, IDs, acronyms, names, version numbers. BM25 is a classic
term-frequency ranking algorithm that scores chunks by exact/near-exact word
overlap with the query, so combining the two catches both cases.

The index is rebuilt from the vector store's own contents (see
``VectorStoreManager.get_all_chunks``) rather than maintained incrementally, so
it can never drift out of sync with what's actually indexed. For a portfolio-
scale document set, rebuilding is fast enough to do on demand.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens; good enough for BM25's bag-of-words model."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class _Entry:
    uid: str
    document: Document


class BM25Index:
    """In-memory BM25 index over a fixed snapshot of chunks."""

    def __init__(self, entries: list[tuple[str, Document]]) -> None:
        self._entries = [_Entry(uid, doc) for uid, doc in entries]
        corpus = [
            tokenize(entry.document.page_content)
            for entry in self._entries
        ]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def build(cls, chunks: list[tuple[str, Document]]) -> "BM25Index":
        return cls(chunks)

    def __len__(self) -> int:
        return len(self._entries)

    def search(
        self,
        query: str,
        *,
        k: int,
        doc_ids: list[str] | None = None,
    ) -> list[tuple[str, Document, float]]:
        """Top-``k`` chunks by BM25 score.

        Returns ``(uid, document, score)`` tuples, best first.

        ``doc_ids`` restricts the search the same way the vector store's
        metadata filter does:
        - ``None`` searches everything
        - an empty list matches nothing
        """
        if self._bm25 is None or k <= 0:
            return []

        if doc_ids is not None and len(doc_ids) == 0:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        allowed = set(doc_ids) if doc_ids is not None else None

        candidate_indices = [
            i
            for i, entry in enumerate(self._entries)
            if allowed is None
            or str(entry.document.metadata.get("doc_id")) in allowed
        ]

        if not candidate_indices:
            return []

        scores = self._bm25.get_scores(query_tokens)

        query_terms = set(query_tokens)

        # BM25 can return 0.0 for a real lexical match when the corpus is
        # very small or a query term occurs in every document. Preserve
        # BM25's ranking, but give genuine lexical matches a positive
        # fallback score so they are not treated as "no match".
        for i in candidate_indices:
            document_terms = set(
                tokenize(self._entries[i].document.page_content)
            )

            overlap = query_terms.intersection(document_terms)

            if overlap and scores[i] <= 0:
                scores[i] = float(len(overlap))

        ranked = sorted(
            candidate_indices,
            key=lambda i: scores[i],
            reverse=True,
        )

        results: list[tuple[str, Document, float]] = []

        for i in ranked:
            document_terms = set(
                tokenize(self._entries[i].document.page_content)
            )

            # Only return documents having at least one exact term
            # in common with the query.
            if query_terms.isdisjoint(document_terms):
                continue

            results.append(
                (
                    self._entries[i].uid,
                    self._entries[i].document,
                    float(scores[i]),
                )
            )

            if len(results) >= k:
                break

        return results


def build_bm25_index(vector_store) -> BM25Index:
    """Fetch every chunk from the vector store and build a fresh BM25 index."""
    try:
        chunks = vector_store.get_all_chunks()
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.exception("Could not build the keyword search index")
        raise VectorStoreError(
            "Could not build the keyword search index."
        ) from exc

    logger.info("BM25 index built: %d chunk(s)", len(chunks))

    return BM25Index.build(chunks)