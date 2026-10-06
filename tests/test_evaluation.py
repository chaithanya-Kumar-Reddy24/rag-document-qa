"""Tests for src/evaluation.py (LLM-judge metrics, dataset CSV parsing, aggregation)."""

from __future__ import annotations

from src.evaluation import (
    EvalCase,
    Evaluator,
    aggregate_scores,
    load_eval_dataset_csv,
    results_to_csv,
)
from src.graph import RAGPipeline
from src.vector_store import VectorStoreManager
from tests.conftest import make_chunk, make_fake_llm


def _index(store: VectorStoreManager) -> None:
    store.add_chunks(
        [
            make_chunk(
                "Supervised learning trains models on labeled data.",
                doc_id="doc1",
                source="ml.txt",
                chunk_id="doc1_01",
                page=1,
            ),
        ]
    )


def _pipeline(retriever, make_generator, make_rewriter, answer: str) -> RAGPipeline:
    generator = make_generator([answer])
    rewriter = make_rewriter(["standalone question"])
    return RAGPipeline(retriever, rewriter, generator, max_context_chars=4000)


def test_load_eval_dataset_csv_parses_rows_and_skips_blank_questions() -> None:
    csv_text = (
        "question,expected_answer,expected_sources\n"
        "What is supervised learning?,Supervised learning uses labeled data,ml.txt;notes.txt\n"
        ",should be skipped,\n"
        "What is unsupervised learning?,,\n"
    )
    cases = load_eval_dataset_csv(csv_text)
    assert len(cases) == 2
    assert cases[0].question == "What is supervised learning?"
    assert cases[0].expected_sources == ("ml.txt", "notes.txt")
    assert cases[1].expected_answer == ""


def test_evaluate_case_computes_retrieval_relevance_without_a_judge_call(
    vector_store, retriever, make_generator, make_rewriter
) -> None:
    _index(vector_store)
    pipeline = _pipeline(retriever, make_generator, make_rewriter, "Supervised learning trains models. [1]")
    judge = make_fake_llm(['{"relevant_blocks": [1]}'])
    evaluator = Evaluator(pipeline, judge)

    result = evaluator.evaluate_case(EvalCase(question="What is supervised learning?"))

    assert result.retrieval_relevance is not None
    assert 0.0 <= result.retrieval_relevance <= 1.0
    assert result.sources == ["ml.txt"]


def test_evaluate_case_context_precision_uses_judge_response(
    vector_store, retriever, make_generator, make_rewriter
) -> None:
    _index(vector_store)
    pipeline = _pipeline(retriever, make_generator, make_rewriter, "Supervised learning trains models. [1]")
    judge = make_fake_llm(['{"relevant_blocks": [1]}', '{"score": 0.9}'])
    evaluator = Evaluator(pipeline, judge)

    result = evaluator.evaluate_case(EvalCase(question="What is supervised learning?"))

    assert result.context_precision == 1.0  # 1 of 1 retrieved block marked relevant


def test_evaluate_case_skips_recall_and_correctness_without_expected_answer(
    vector_store, retriever, make_generator, make_rewriter
) -> None:
    _index(vector_store)
    pipeline = _pipeline(retriever, make_generator, make_rewriter, "Supervised learning trains models. [1]")
    judge = make_fake_llm(['{"relevant_blocks": [1]}', '{"score": 0.8}'])
    evaluator = Evaluator(pipeline, judge)

    result = evaluator.evaluate_case(EvalCase(question="What is supervised learning?"))

    assert result.context_recall is None
    assert result.answer_correctness is None
    assert any("expected_answer" in note for note in result.notes)


def test_evaluate_case_computes_recall_and_correctness_when_expected_answer_given(
    vector_store, retriever, make_generator, make_rewriter
) -> None:
    _index(vector_store)
    pipeline = _pipeline(retriever, make_generator, make_rewriter, "Supervised learning trains models. [1]")
    judge = make_fake_llm(
        ['{"relevant_blocks": [1]}', '{"score": 0.8}', '{"score": 0.95}', '{"score": 0.7}']
    )
    evaluator = Evaluator(pipeline, judge)

    result = evaluator.evaluate_case(
        EvalCase(question="What is supervised learning?", expected_answer="It trains on labeled data.")
    )

    assert result.context_recall == 0.95
    assert result.answer_correctness == 0.7


def test_groundedness_is_none_when_pipeline_returns_not_found(
    vector_store, retriever, make_generator, make_rewriter
) -> None:
    # empty store -> no context -> pipeline returns the not-found message, no LLM answer call
    pipeline = _pipeline(retriever, make_generator, make_rewriter, "THIS SHOULD NEVER APPEAR")
    judge = make_fake_llm(['{"relevant_blocks": []}'])
    evaluator = Evaluator(pipeline, judge)

    result = evaluator.evaluate_case(EvalCase(question="What is supervised learning?"))

    assert result.groundedness is None
    assert result.context_precision is None  # no retrieved blocks to judge


def test_judge_returning_unparseable_json_yields_none_not_zero(
    vector_store, retriever, make_generator, make_rewriter
) -> None:
    _index(vector_store)
    pipeline = _pipeline(retriever, make_generator, make_rewriter, "Supervised learning trains models. [1]")
    judge = make_fake_llm(["not valid json at all"])
    evaluator = Evaluator(pipeline, judge)

    result = evaluator.evaluate_case(EvalCase(question="What is supervised learning?"))

    assert result.context_precision is None  # not 0.0 - the judge call failed, not "no relevant blocks"


def test_aggregate_scores_ignores_none_values() -> None:
    from src.evaluation import EvalResult

    results = [
        EvalResult(
            question="q1",
            expected_answer="",
            generated_answer="a1",
            search_query="q1",
            context="c1",
            sources=[],
            retrieval_relevance=0.8,
            groundedness=0.9,
        ),
        EvalResult(
            question="q2",
            expected_answer="",
            generated_answer="a2",
            search_query="q2",
            context="c2",
            sources=[],
            retrieval_relevance=None,
            groundedness=0.5,
        ),
    ]
    aggregates = aggregate_scores(results)
    assert aggregates["retrieval_relevance"] == 0.8  # only one non-None value
    assert aggregates["groundedness"] == 0.7  # mean of 0.9 and 0.5
    assert aggregates["context_precision"] is None  # no values at all


def test_results_to_csv_round_trips_basic_fields() -> None:
    from src.evaluation import EvalResult

    results = [
        EvalResult(
            question="What is X?",
            expected_answer="X is Y",
            generated_answer="X is Y.",
            search_query="What is X?",
            context="[1] ...",
            sources=["a.txt"],
            retrieval_relevance=0.7,
            answer_correctness=0.9,
            latency_seconds=1.234,
        )
    ]
    csv_text = results_to_csv(results)
    assert "What is X?" in csv_text
    assert "a.txt" in csv_text
    assert "0.7" in csv_text


def test_compare_retrieval_strategies_runs_every_pipeline_and_returns_per_strategy_results(
    vector_store, retriever, make_generator, make_rewriter
) -> None:
    from src.evaluation import compare_retrieval_strategies

    _index(vector_store)
    pipeline_a = _pipeline(retriever, make_generator, make_rewriter, "Answer A. [1]")
    pipeline_b = _pipeline(retriever, make_generator, make_rewriter, "Answer B. [1]")
    judge = make_fake_llm(
        [
            '{"relevant_blocks": [1]}',  # pipeline_a: context_precision
            '{"score": 0.9}',  # pipeline_a: groundedness
            '{"relevant_blocks": [1]}',  # pipeline_b: context_precision
            '{"score": 0.9}',  # pipeline_b: groundedness
        ]
    )

    results = compare_retrieval_strategies(
        [EvalCase(question="What is supervised learning?")],
        {"Strategy A": pipeline_a, "Strategy B": pipeline_b},
        judge,
    )

    assert set(results) == {"Strategy A", "Strategy B"}
    assert results["Strategy A"][0].generated_answer == "Answer A. [1]"
    assert results["Strategy B"][0].generated_answer == "Answer B. [1]"


def test_compare_retrieval_strategies_reports_progress_per_strategy(
    vector_store, retriever, make_generator, make_rewriter
) -> None:
    from src.evaluation import compare_retrieval_strategies

    _index(vector_store)
    pipeline_a = _pipeline(retriever, make_generator, make_rewriter, "Answer A. [1]")
    judge = make_fake_llm(['{"relevant_blocks": [1]}', '{"score": 0.9}'])

    calls: list[tuple[int, int, int, int]] = []
    compare_retrieval_strategies(
        [EvalCase(question="What is supervised learning?")],
        {"Only strategy": pipeline_a},
        judge,
        on_progress=lambda *args: calls.append(args),
    )

    assert calls == [(1, 1, 1, 1)]


def test_comparison_table_has_one_row_per_metric_plus_latency() -> None:
    from src.evaluation import EvalResult, comparison_table

    results_by_strategy = {
        "Vector": [
            EvalResult(
                question="q",
                expected_answer="",
                generated_answer="a",
                search_query="q",
                context="c",
                sources=[],
                retrieval_relevance=0.5,
                latency_seconds=1.0,
            )
        ],
        "Hybrid": [
            EvalResult(
                question="q",
                expected_answer="",
                generated_answer="a",
                search_query="q",
                context="c",
                sources=[],
                retrieval_relevance=0.8,
                latency_seconds=2.0,
            )
        ],
    }
    table = comparison_table(results_by_strategy)
    metrics = {row["metric"] for row in table}
    assert "Retrieval relevance" in metrics
    assert "Avg. latency (s)" in metrics

    relevance_row = next(row for row in table if row["metric"] == "Retrieval relevance")
    assert relevance_row["Vector"] == 0.5
    assert relevance_row["Hybrid"] == 0.8

    latency_row = next(row for row in table if row["metric"] == "Avg. latency (s)")
    assert latency_row["Vector"] == 1.0
    assert latency_row["Hybrid"] == 2.0
