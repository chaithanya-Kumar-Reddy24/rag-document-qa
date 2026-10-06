"""Tests for src/reranker.py (LLM-based re-ranking, with a fake LLM)."""

from __future__ import annotations

from src.reranker import LLMReranker
from tests.conftest import make_chunk, make_fake_llm


def _candidates() -> list:
    return [
        make_chunk("Python is popular for data science.", doc_id="d1", source="a.txt", chunk_id="c1"),
        make_chunk("Bananas are a good source of potassium.", doc_id="d2", source="b.txt", chunk_id="c1"),
    ]


def test_rerank_applies_judge_scores_and_sorts_descending() -> None:
    reranker = LLMReranker(make_fake_llm(['{"scores": [{"block": 1, "score": 0.2}, {"block": 2, "score": 0.9}]}']))
    result = reranker.rerank("What is a good source of potassium?", _candidates())

    assert [doc.metadata["relevance_score"] for doc in result] == [0.9, 0.2]
    assert "potassium" in result[0].page_content.lower()


def test_rerank_on_empty_candidates_returns_empty() -> None:
    reranker = LLMReranker(make_fake_llm(["irrelevant"]))
    assert reranker.rerank("anything", []) == []


def test_rerank_falls_back_to_fusion_score_on_unparseable_json() -> None:
    candidates = _candidates()
    candidates[0].metadata["fusion_score"] = 0.7
    candidates[1].metadata["fusion_score"] = 0.3

    reranker = LLMReranker(make_fake_llm(["not valid json"]))
    result = reranker.rerank("anything", candidates)

    scores = {doc.metadata["source"]: doc.metadata["relevance_score"] for doc in result}
    assert scores["a.txt"] == 0.7
    assert scores["b.txt"] == 0.3


def test_rerank_falls_back_for_missing_block_numbers() -> None:
    # judge only scores block 1; block 2 should fall back to its fusion_score (default 0.0)
    reranker = LLMReranker(make_fake_llm(['{"scores": [{"block": 1, "score": 0.5}]}']))
    result = reranker.rerank("anything", _candidates())

    by_source = {doc.metadata["source"]: doc.metadata["relevance_score"] for doc in result}
    assert by_source["a.txt"] == 0.5
    assert by_source["b.txt"] == 0.0


def test_rerank_clamps_out_of_range_scores() -> None:
    reranker = LLMReranker(make_fake_llm(['{"scores": [{"block": 1, "score": 5.0}, {"block": 2, "score": -1.0}]}']))
    result = reranker.rerank("anything", _candidates())

    scores = {doc.metadata["source"]: doc.metadata["relevance_score"] for doc in result}
    assert scores["a.txt"] == 1.0
    assert scores["b.txt"] == 0.0
