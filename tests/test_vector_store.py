"""Tests for src/vector_store.py (indexing, duplicates, filtering, deletion)."""

from __future__ import annotations

from src.text_processor import split_documents
from src.vector_store import VectorStoreManager
from tests.conftest import make_chunk


def _index_sample(store: VectorStoreManager, doc_id: str = "doc1", filename: str = "a.txt") -> None:
    chunks = [
        make_chunk("Python is a programming language.", doc_id=doc_id, source=filename, chunk_id=f"{doc_id}_01"),
        make_chunk("Machine learning uses data to train models.", doc_id=doc_id, source=filename, chunk_id=f"{doc_id}_02"),
    ]
    store.add_chunks(chunks)


def test_add_and_count_chunks(vector_store: VectorStoreManager) -> None:
    _index_sample(vector_store)
    assert vector_store.count_chunks() == 2


def test_has_document_detects_duplicates(vector_store: VectorStoreManager) -> None:
    _index_sample(vector_store, doc_id="doc1")
    assert vector_store.has_document("doc1") is True
    assert vector_store.has_document("doc-not-indexed") is False


def test_list_documents_aggregates_chunk_counts(vector_store: VectorStoreManager) -> None:
    _index_sample(vector_store, doc_id="doc1", filename="a.txt")
    _index_sample(vector_store, doc_id="doc2", filename="b.txt")
    documents = vector_store.list_documents()
    assert {doc.doc_id for doc in documents} == {"doc1", "doc2"}
    assert all(doc.chunks == 2 for doc in documents)


def test_search_returns_relevant_chunk_first(vector_store: VectorStoreManager) -> None:
    _index_sample(vector_store)
    results = vector_store.search("Tell me about Python programming", k=2)
    assert results, "expected at least one search result"
    top_doc, top_score = results[0]
    assert "python" in top_doc.page_content.lower()
    assert 0.0 <= top_score <= 1.0


def test_search_with_doc_id_filter_restricts_results(vector_store: VectorStoreManager) -> None:
    _index_sample(vector_store, doc_id="doc1", filename="a.txt")
    _index_sample(vector_store, doc_id="doc2", filename="b.txt")
    results = vector_store.search("Python", k=10, doc_ids=["doc1"])
    assert all(doc.metadata["doc_id"] == "doc1" for doc, _ in results)


def test_search_with_empty_doc_id_list_returns_nothing(vector_store: VectorStoreManager) -> None:
    _index_sample(vector_store)
    assert vector_store.search("Python", k=10, doc_ids=[]) == []


def test_search_on_empty_store_returns_empty_list(vector_store: VectorStoreManager) -> None:
    assert vector_store.search("anything", k=4) == []


def test_delete_document_removes_only_its_chunks(vector_store: VectorStoreManager) -> None:
    _index_sample(vector_store, doc_id="doc1", filename="a.txt")
    _index_sample(vector_store, doc_id="doc2", filename="b.txt")
    vector_store.delete_document("doc1")
    remaining = vector_store.list_documents()
    assert [doc.doc_id for doc in remaining] == ["doc2"]


def test_clear_empties_the_collection(vector_store: VectorStoreManager) -> None:
    _index_sample(vector_store)
    vector_store.clear()
    assert vector_store.count_chunks() == 0
