"""Observability: per-query latency breakdown and token/cost estimates.

Every query already flows through a multi-stage pipeline (query rewriting,
vector search, keyword search, fusion, re-ranking, generation). This module
gives that pipeline a "where did the time and money go?" answer, grounded in
actual measurements taken during the run — not estimates of what a stage
*should* cost.

Two independent things live here:

- ``RetrievalTimings`` — wall-clock milliseconds per retrieval sub-stage,
  populated by ``DocumentRetriever``/``HybridRetriever`` on every call to
  ``retrieve()`` as ``self.last_timings``, and read by the LangGraph
  ``retrieve_documents`` node.
- ``TokenUsageCallback`` + ``estimate_cost_usd`` — token counts read from the
  LLM response's usage metadata, and a rough cost estimate from a small,
  hand-maintained pricing table.

The cost table is the one place in this module that can silently go stale:
OpenAI's prices change, and this project's knowledge of them is frozen at
whenever it was last edited. Treat every number ``estimate_cost_usd`` returns
as an order-of-magnitude estimate, not a bill — and check
https://openai.com/api/pricing/ before using it for real budgeting. A model
not in the table returns ``None`` rather than a guessed price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class RetrievalTimings:
    """Per-stage retrieval timing, in milliseconds. ``None`` means that stage
    doesn't apply to the retriever that produced this (e.g. ``keyword_ms`` is
    ``None`` for vector-only retrieval; ``rerank_ms`` is ``None`` when hybrid
    retrieval ran without the LLM re-ranker)."""

    vector_ms: float = 0.0
    keyword_ms: Optional[float] = None
    fusion_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    total_ms: float = 0.0


class TokenUsageCallback(BaseCallbackHandler):
    """Accumulates prompt/completion/total token counts across every LLM call
    it's attached to (via ``config={"callbacks": [...]}}`` on an ``.invoke()``).

    Reads ``response.llm_output["token_usage"]``, which ``ChatOpenAI`` populates
    for ordinary (non-streaming) calls, falling back to each message's
    ``usage_metadata`` when present. Streaming calls may not populate either,
    depending on the langchain-openai version and whether ``stream_usage`` is
    enabled on the model — when neither is available, counts simply stay at 0
    and callers should treat that as "unknown", not "zero cost".
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        usage: dict[str, int] = {}
        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage") or {}

        if not usage:
            for generation in getattr(response, "generations", None) or []:
                for item in generation:
                    message = getattr(item, "message", None)
                    meta = getattr(message, "usage_metadata", None) if message else None
                    if meta:
                        usage = {
                            "prompt_tokens": meta.get("input_tokens", 0),
                            "completion_tokens": meta.get("output_tokens", 0),
                            "total_tokens": meta.get("total_tokens", 0),
                        }
                        break
                if usage:
                    break

        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)
        self.call_count += 1


# Illustrative USD price per 1M tokens (input, output) - NOT a live pricing feed.
# See the module docstring: verify at https://openai.com/api/pricing/ before
# relying on this for anything beyond a rough order-of-magnitude estimate.
_PRICING_PER_MILLION_TOKENS_USD: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Rough cost estimate for one call in USD, or ``None`` if ``model`` isn't in
    the pricing table above — a missing price is reported as unknown, never as
    free, so it can't be misread as "this call cost nothing"."""
    prices = _PRICING_PER_MILLION_TOKENS_USD.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return round((prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000, 6)
