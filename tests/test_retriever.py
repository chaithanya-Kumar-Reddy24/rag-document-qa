"""Tests for src/retriever.py (top-k and relevance threshold behaviour)."""

from __future__ import annotations

from src.retriever import DocumentRetriever
from src.vector_store import VectorStoreManager
from tests.conftest import make_chunk


def _index(store: VectorStoreManager) -> None:
    store.add_chunks(
        [
            make_chunk(
                "Python is a popular programming language for data science.",
                doc_id="doc1",
                source="python.txt",
                chunk_id="doc1_01",
            ),
            make_chunk(
                "Bananas and apples are healthy fruits to eat daily.",
                doc_id="doc2",
                source="fruit.txt",
                chunk_id="doc2_01",
            ),
        ]
    )


def test_retrieve_returns_relevant_chunk_with_score(vector_store: VectorStoreManager) -> None:
    _index(vector_store)
    retriever = DocumentRetriever(vector_store, default_top_k=4, default_threshold=0.05)
    results = retriever.retrieve("What programming language is popular?")
    assert results
    assert "python" in results[0].page_content.lower()
    assert "relevance_score" in results[0].metadata


def test_retrieve_respects_high_threshold(vector_store: VectorStoreManager) -> None:
    _index(vector_store)
    retriever = DocumentRetriever(vector_store, default_top_k=4, default_threshold=0.05)
    results = retriever.retrieve("completely unrelated topic about spacecraft engines", threshold=0.99)
    assert results == []


def test_retrieve_respects_top_k(vector_store: VectorStoreManager) -> None:
    _index(vector_store)
    retriever = DocumentRetriever(vector_store, default_top_k=4, default_threshold=0.0)
    results = retriever.retrieve("fruit and programming", top_k=1)
    assert len(results) <= 1


def test_retrieve_on_empty_store_returns_empty(vector_store: VectorStoreManager) -> None:
    retriever = DocumentRetriever(vector_store, default_top_k=4, default_threshold=0.0)
    assert retriever.retrieve("anything") == []


# --------------------------------------------------------------------------- #
# HybridRetriever (vector + BM25 fusion + LLM re-ranking)
# --------------------------------------------------------------------------- #


def _hybrid(vector_store: VectorStoreManager, judge_responses: list[str], **kwargs) -> "HybridRetriever":
    from src.keyword_search import build_bm25_index
    from src.reranker import LLMReranker
    from src.retriever import HybridRetriever
    from tests.conftest import make_fake_llm

    keyword_index = build_bm25_index(vector_store)
    reranker = LLMReranker(make_fake_llm(judge_responses))
    defaults = dict(default_top_k=4, default_threshold=0.0, candidate_pool_size=10, rrf_k=60)
    defaults.update(kwargs)
    return HybridRetriever(vector_store, reranker, keyword_index_provider=lambda: keyword_index, **defaults)


def test_hybrid_retrieve_combines_and_reranks(vector_store: VectorStoreManager) -> None:
    _index(vector_store)
    retriever = _hybrid(
        vector_store, ['{"scores": [{"block": 1, "score": 0.9}, {"block": 2, "score": 0.1}]}']
    )
    results = retriever.retrieve("What programming language is popular?")
    assert results
    assert results[0].metadata["relevance_score"] == 0.9


def test_hybrid_retrieve_respects_threshold(vector_store: VectorStoreManager) -> None:
    _index(vector_store)
    retriever = _hybrid(
        vector_store,
        ['{"scores": [{"block": 1, "score": 0.9}, {"block": 2, "score": 0.1}]}'],
        default_threshold=0.5,
    )
    results = retriever.retrieve("What programming language is popular?")
    assert all(doc.metadata["relevance_score"] >= 0.5 for doc in results)
    assert len(results) == 1


def test_hybrid_retrieve_respects_top_k(vector_store: VectorStoreManager) -> None:
    _index(vector_store)
    retriever = _hybrid(
        vector_store,
        ['{"scores": [{"block": 1, "score": 0.9}, {"block": 2, "score": 0.8}]}'],
        default_top_k=1,
        default_threshold=0.0,
    )
    results = retriever.retrieve("programming and fruit")
    assert len(results) <= 1


def test_hybrid_retrieve_with_empty_doc_id_list_returns_nothing(vector_store: VectorStoreManager) -> None:
    _index(vector_store)
    retriever = _hybrid(vector_store, ["irrelevant"])
    assert retriever.retrieve("Python", doc_ids=[]) == []


def test_hybrid_retrieve_on_empty_store_returns_empty(vector_store: VectorStoreManager) -> None:
    retriever = _hybrid(vector_store, ["irrelevant"])
    assert retriever.retrieve("anything") == []


def test_hybrid_retrieve_survives_reranker_failure(vector_store: VectorStoreManager) -> None:
    # unparseable judge output -> falls back to fusion order instead of crashing
    _index(vector_store)
    retriever = _hybrid(vector_store, ["not valid json"], default_threshold=0.0)
    results = retriever.retrieve("What programming language is popular?")
    assert results  # still returns something, scored by fusion fallback


def test_hybrid_retrieve_without_reranking_uses_normalized_fusion_scores(
    vector_store: VectorStoreManager,
) -> None:
    from src.keyword_search import build_bm25_index
    from src.retriever import HybridRetriever

    _index(vector_store)
    keyword_index = build_bm25_index(vector_store)
    retriever = HybridRetriever(
        vector_store,
        reranker=None,
        keyword_index_provider=lambda: keyword_index,
        default_top_k=4,
        default_threshold=0.0,
        candidate_pool_size=10,
        enable_reranking=False,
    )
    results = retriever.retrieve("What programming language is popular?")
    assert results
    # normalized scores should land in [0, 1], with the top candidate at 1.0
    assert all(0.0 <= doc.metadata["relevance_score"] <= 1.0 for doc in results)
    assert results[0].metadata["relevance_score"] == 1.0


def test_hybrid_retrieve_enable_reranking_false_never_calls_the_reranker(
    vector_store: VectorStoreManager, make_reranker
) -> None:
    from src.keyword_search import build_bm25_index
    from src.retriever import HybridRetriever

    _index(vector_store)
    keyword_index = build_bm25_index(vector_store)
    # a reranker whose only response is a marker that would prove it was called
    reranker = make_reranker(['{"scores": [{"block": 1, "score": 0.999}]}'])
    retriever = HybridRetriever(
        vector_store,
        reranker,
        keyword_index_provider=lambda: keyword_index,
        default_top_k=4,
        default_threshold=0.0,
        candidate_pool_size=10,
        enable_reranking=False,
    )
    results = retriever.retrieve("What programming language is popular?")
    # if the reranker had run, the top score would be exactly 0.999
    assert results[0].metadata["relevance_score"] != 0.999


# --------------------------------------------------------------------------- #
# last_timings (observability)
# --------------------------------------------------------------------------- #


def test_document_retriever_records_last_timings(vector_store: VectorStoreManager) -> None:
    _index(vector_store)
    retriever = DocumentRetriever(vector_store, default_top_k=4, default_threshold=0.0)
    retriever.retrieve("Python programming")

    assert retriever.last_timings.vector_ms >= 0.0
    assert retriever.last_timings.total_ms >= 0.0
    assert retriever.last_timings.keyword_ms is None


def test_hybrid_retriever_records_granular_last_timings_with_reranking(
    vector_store: VectorStoreManager,
) -> None:
    _index(vector_store)
    retriever = _hybrid(
        vector_store, ['{"scores": [{"block": 1, "score": 0.9}, {"block": 2, "score": 0.1}]}']
    )
    retriever.retrieve("Python programming")

    timings = retriever.last_timings
    assert timings.vector_ms >= 0.0
    assert timings.keyword_ms is not None and timings.keyword_ms >= 0.0
    assert timings.fusion_ms is not None and timings.fusion_ms >= 0.0
    assert timings.rerank_ms is not None and timings.rerank_ms >= 0.0
    assert timings.total_ms >= 0.0


def test_hybrid_retriever_rerank_ms_is_none_when_reranking_disabled(
    vector_store: VectorStoreManager,
) -> None:
    from src.keyword_search import build_bm25_index
    from src.retriever import HybridRetriever

    _index(vector_store)
    keyword_index = build_bm25_index(vector_store)
    retriever = HybridRetriever(
        vector_store,
        reranker=None,
        keyword_index_provider=lambda: keyword_index,
        default_top_k=4,
        default_threshold=0.0,
        candidate_pool_size=10,
        enable_reranking=False,
    )
    retriever.retrieve("Python programming")

    assert retriever.last_timings.rerank_ms is None
