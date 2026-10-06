"""Tests for src/observability.py (token usage callback, cost estimation)."""

from __future__ import annotations

from types import SimpleNamespace

from src.observability import RetrievalTimings, TokenUsageCallback, estimate_cost_usd


def test_retrieval_timings_defaults() -> None:
    timings = RetrievalTimings()
    assert timings.vector_ms == 0.0
    assert timings.keyword_ms is None
    assert timings.rerank_ms is None


def test_token_usage_callback_reads_llm_output_token_usage() -> None:
    callback = TokenUsageCallback()
    response = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
        generations=[],
    )
    callback.on_llm_end(response)

    assert callback.prompt_tokens == 100
    assert callback.completion_tokens == 20
    assert callback.total_tokens == 120
    assert callback.call_count == 1


def test_token_usage_callback_accumulates_across_multiple_calls() -> None:
    callback = TokenUsageCallback()
    response = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        generations=[],
    )
    callback.on_llm_end(response)
    callback.on_llm_end(response)

    assert callback.prompt_tokens == 20
    assert callback.total_tokens == 30
    assert callback.call_count == 2


def test_token_usage_callback_falls_back_to_message_usage_metadata() -> None:
    callback = TokenUsageCallback()
    message = SimpleNamespace(usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
    generation = [SimpleNamespace(message=message)]
    response = SimpleNamespace(llm_output=None, generations=[generation])
    callback.on_llm_end(response)

    assert callback.prompt_tokens == 7
    assert callback.completion_tokens == 3
    assert callback.total_tokens == 10


def test_token_usage_callback_handles_missing_usage_gracefully() -> None:
    callback = TokenUsageCallback()
    response = SimpleNamespace(llm_output=None, generations=[])
    callback.on_llm_end(response)

    assert callback.prompt_tokens == 0
    assert callback.completion_tokens == 0
    assert callback.total_tokens == 0
    assert callback.call_count == 1


def test_estimate_cost_usd_for_known_model() -> None:
    # gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output
    cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == 0.75


def test_estimate_cost_usd_for_unknown_model_returns_none() -> None:
    assert estimate_cost_usd("some-future-model-xyz", prompt_tokens=1000, completion_tokens=200) is None


def test_estimate_cost_usd_zero_tokens_is_zero_not_none_for_known_model() -> None:
    assert estimate_cost_usd("gpt-4o-mini", prompt_tokens=0, completion_tokens=0) == 0.0
