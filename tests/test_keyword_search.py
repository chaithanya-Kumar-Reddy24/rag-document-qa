"""Tests for src/keyword_search.py (BM25 index)."""

from __future__ import annotations

from src.keyword_search import BM25Index, build_bm25_index, tokenize
from tests.conftest import make_chunk


def _entries() -> list[tuple[str, object]]:
    return [
        ("doc1:c1", make_chunk("Python is a popular programming language.", doc_id="doc1", source="a.txt")),
        ("doc2:c1", make_chunk("Bananas and apples are healthy fruits.", doc_id="doc2", source="b.txt")),
    ]


def test_tokenize_lowercases_and_splits_on_words() -> None:
    assert tokenize("Hello, World! v2.0") == ["hello", "world", "v2", "0"]


def test_bm25_search_finds_term_overlap() -> None:
    index = BM25Index.build(_entries())
    results = index.search("popular Python language", k=5)
    assert results
    uid, doc, score = results[0]
    assert uid == "doc1:c1"
    assert score > 0


def test_bm25_search_excludes_zero_score_matches() -> None:
    index = BM25Index.build(_entries())
    results = index.search("completely unrelated spacecraft terminology", k=5)
    assert results == []


def test_bm25_search_respects_doc_id_filter() -> None:
    index = BM25Index.build(_entries())
    results = index.search("fruit", k=5, doc_ids=["doc1"])
    assert results == []  # doc1 doesn't mention fruit, and doc2 is filtered out


def test_bm25_search_with_empty_doc_id_list_returns_nothing() -> None:
    index = BM25Index.build(_entries())
    assert index.search("python", k=5, doc_ids=[]) == []


def test_bm25_index_on_empty_corpus_returns_nothing() -> None:
    index = BM25Index.build([])
    assert len(index) == 0
    assert index.search("anything", k=5) == []


def test_build_bm25_index_reads_from_vector_store(vector_store) -> None:
    vector_store.add_chunks(
        [make_chunk("Python is great for data science.", doc_id="doc1", source="a.txt", chunk_id="c1")]
    )
    index = build_bm25_index(vector_store)
    results = index.search("Python data science", k=5)
    assert results
