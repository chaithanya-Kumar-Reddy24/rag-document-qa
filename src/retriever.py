"""Retrievers: semantic search + relevance threshold, and hybrid (vector + BM25 +
LLM re-ranking) search.

Only chunks whose final relevance score reaches the threshold are returned, so
weak matches never reach the LLM (the first line of defence against
hallucination: no context, no answer).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Sequence

from langchain_core.documents import Document

from src.keyword_search import BM25Index
from src.observability import RetrievalTimings
from src.reranker import LLMReranker
from src.utils import chunk_uid
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Finds the most relevant chunks for a standalone search query using vector
    search alone (cosine similarity threshold)."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        default_top_k: int = 4,
        default_threshold: float = 0.3,
    ) -> None:
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._default_threshold = default_threshold
        self.last_timings = RetrievalTimings()

    def retrieve(
        self,
        query: str,
        *,
        doc_ids: Sequence[str] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[Document]:
        """Return up to ``top_k`` chunks scoring at least ``threshold`` (best first).

        Each returned chunk carries its score in ``metadata['relevance_score']``.
        """
        k = top_k or self._default_top_k
        minimum = self._default_threshold if threshold is None else threshold

        started = time.perf_counter()
        scored = self._vector_store.search(query, k=k, doc_ids=doc_ids)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.last_timings = RetrievalTimings(vector_ms=round(elapsed_ms, 1), total_ms=round(elapsed_ms, 1))

        kept = sorted(
            ((doc, score) for doc, score in scored if score >= minimum),
            key=lambda pair: pair[1],
            reverse=True,
        )
        logger.info(
            "Retrieved %d chunk(s) (%d above threshold %.2f, k=%d)",
            len(scored),
            len(kept),
            minimum,
            k,
        )
        return [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": round(score, 4)},
            )
            for doc, score in kept
        ]


class HybridRetriever:
    """Vector search + BM25 keyword search, fused with Reciprocal Rank Fusion (RRF),
    then filtered by threshold — optionally re-scored by an LLM re-ranker first.

        query
          |
          +--> vector search (cosine similarity)  --+
          |                                          |--> RRF fusion --> candidate pool
          +--> BM25 keyword search                 --+
                                                            |
                                                 LLM re-ranker (if enable_reranking)
                                                            |
                                                 threshold filter -> top_k

    RRF combines two differently-scaled rankings (cosine similarity vs. BM25 score)
    by rank position rather than raw score, which avoids having to calibrate the two
    scales against each other: ``fusion_score(d) = sum(1 / (rrf_k + rank_in_list))``
    over every list ``d`` appears in.

    With ``enable_reranking=False``, fusion scores are min-max normalized and used
    directly as ``relevance_score`` — this is the "Hybrid" (no re-ranker) leg of the
    Evaluation Dashboard's retrieval-strategy comparison; ``enable_reranking=True`` is the optional "Hybrid + Reranker" leg.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        reranker: Optional[LLMReranker],
        keyword_index_provider: Callable[[], Optional[BM25Index]],
        default_top_k: int = 4,
        default_threshold: float = 0.3,
        candidate_pool_size: int = 20,
        rrf_k: int = 60,
        enable_reranking: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._reranker = reranker
        self._keyword_index_provider = keyword_index_provider
        self._default_top_k = default_top_k
        self._default_threshold = default_threshold
        self._candidate_pool_size = candidate_pool_size
        self._rrf_k = rrf_k
        self._enable_reranking = enable_reranking and reranker is not None
        self.last_timings = RetrievalTimings()

    def _fuse(
        self,
        vector_results: list[tuple[Document, float]],
        keyword_results: list[tuple[str, Document, float]],
    ) -> list[Document]:
        """Reciprocal Rank Fusion over the two ranked lists, deduplicated by chunk id."""
        fusion_scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        debug: dict[str, dict[str, float]] = {}

        for rank, (doc, score) in enumerate(vector_results, start=1):
            uid = chunk_uid(doc.metadata)
            fusion_scores[uid] = fusion_scores.get(uid, 0.0) + 1.0 / (self._rrf_k + rank)
            docs.setdefault(uid, doc)
            debug.setdefault(uid, {})["vector_score"] = round(score, 4)

        for rank, (uid, doc, score) in enumerate(keyword_results, start=1):
            fusion_scores[uid] = fusion_scores.get(uid, 0.0) + 1.0 / (self._rrf_k + rank)
            docs.setdefault(uid, doc)
            debug.setdefault(uid, {})["keyword_score"] = round(score, 4)

        ranked_uids = sorted(fusion_scores, key=lambda uid: fusion_scores[uid], reverse=True)
        fused: list[Document] = []
        for uid in ranked_uids:
            source = docs[uid]
            metadata = {**source.metadata, "fusion_score": round(fusion_scores[uid], 6), **debug[uid]}
            fused.append(Document(page_content=source.page_content, metadata=metadata))
        return fused

    @staticmethod
    def _normalize_fusion_scores(candidates: list[Document]) -> list[Document]:
        """Min-max scale ``fusion_score`` to 0-1 as ``relevance_score``, so the
        threshold/top_k logic works the same whether or not re-ranking ran.

        RRF scores are small and clustered (typically well under 0.1 with the
        default ``rrf_k``), so used directly they'd fail almost any relevance
        threshold tuned for the 0-1 range the LLM re-ranker and cosine similarity
        both use. Min-max scaling makes them comparable on that same 0-1 scale
        without claiming a false absolute meaning for the number.
        """
        scores = [doc.metadata.get("fusion_score", 0.0) for doc in candidates]
        low, high = min(scores), max(scores)
        spread = high - low

        scaled: list[Document] = []
        for doc, score in zip(candidates, scores):
            normalized = 1.0 if spread == 0 else (score - low) / spread
            metadata = {**doc.metadata, "relevance_score": round(normalized, 4)}
            scaled.append(Document(page_content=doc.page_content, metadata=metadata))
        return scaled

    def retrieve(
        self,
        query: str,
        *,
        doc_ids: Sequence[str] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[Document]:
        k = top_k or self._default_top_k
        minimum = self._default_threshold if threshold is None else threshold
        pool_size = max(self._candidate_pool_size, k)
        doc_id_list = list(doc_ids) if doc_ids is not None else None
        overall_start = time.perf_counter()

        if doc_id_list is not None and len(doc_id_list) == 0:
            self.last_timings = RetrievalTimings(vector_ms=0.0, keyword_ms=0.0, fusion_ms=0.0, total_ms=0.0)
            return []

        t0 = time.perf_counter()
        vector_results = self._vector_store.search(query, k=pool_size, doc_ids=doc_id_list)
        t1 = time.perf_counter()

        keyword_index = self._keyword_index_provider()
        keyword_results = (
            keyword_index.search(query, k=pool_size, doc_ids=doc_id_list) if keyword_index else []
        )
        t2 = time.perf_counter()

        candidates = self._fuse(vector_results, keyword_results)[:pool_size]
        t3 = time.perf_counter()

        if not candidates:
            logger.info("Hybrid retrieval found no candidates")
            self.last_timings = RetrievalTimings(
                vector_ms=round((t1 - t0) * 1000, 1),
                keyword_ms=round((t2 - t1) * 1000, 1),
                fusion_ms=round((t3 - t2) * 1000, 1),
                total_ms=round((time.perf_counter() - overall_start) * 1000, 1),
            )
            return []

        scored = (
            self._reranker.rerank(query, candidates)
            if self._enable_reranking
            else self._normalize_fusion_scores(candidates)
        )
        t4 = time.perf_counter()
        kept = [doc for doc in scored if doc.metadata.get("relevance_score", 0.0) >= minimum][:k]

        self.last_timings = RetrievalTimings(
            vector_ms=round((t1 - t0) * 1000, 1),
            keyword_ms=round((t2 - t1) * 1000, 1),
            fusion_ms=round((t3 - t2) * 1000, 1),
            rerank_ms=round((t4 - t3) * 1000, 1) if self._enable_reranking else None,
            total_ms=round((time.perf_counter() - overall_start) * 1000, 1),
        )

        logger.info(
            "Hybrid retrieval (%s): %d vector + %d keyword -> %d fused candidate(s) -> "
            "%d above threshold %.2f (k=%d)",
            "reranked" if self._enable_reranking else "fusion-only",
            len(vector_results),
            len(keyword_results),
            len(candidates),
            len(kept),
            minimum,
            k,
        )
        return kept
