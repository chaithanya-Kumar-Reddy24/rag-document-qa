"""LLM-based re-ranking: the last-mile relevance check after hybrid retrieval.

Vector and keyword search are both fast but approximate; each returns a fairly
large candidate pool (e.g. 20 chunks) that's ranked by cheap proxies (cosine
similarity, term overlap). A re-ranker looks at each candidate *together with
the actual question* and scores how useful it would really be for answering
it, which catches cases where a chunk is topically close but not actually
useful, or phrased so differently that surface signals underrated it.

This project uses the LLM itself as the re-ranker (a common "listwise LLM
re-ranking" pattern) instead of a dedicated cross-encoder model, which would
add a heavy ML dependency (e.g. sentence-transformers + torch) for a
portfolio-scale project. The trade-off is one extra LLM call per question.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.prompts import RERANK_PROMPT

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_scores(raw: str, count: int) -> dict[int, float] | None:
    cleaned = _JSON_FENCE.sub("", raw.strip()).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    entries = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return None

    scores: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        block, score = entry.get("block"), entry.get("score")
        if isinstance(block, int) and isinstance(score, (int, float)) and 1 <= block <= count:
            scores[block] = max(0.0, min(1.0, float(score)))
    return scores or None


class LLMReranker:
    """Scores each candidate chunk's relevance to the query with the LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = RERANK_PROMPT | llm | StrOutputParser()

    def rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        """Return the candidates with ``metadata['relevance_score']`` set to the
        re-ranker's score, sorted best-first. Falls back to the incoming fusion
        order (unscored) if the LLM call or its JSON response can't be used, so a
        re-ranking hiccup degrades retrieval quality rather than breaking it.
        """
        if not candidates:
            return []

        numbered = "\n\n".join(
            f"[{i}]\n{doc.page_content}" for i, doc in enumerate(candidates, start=1)
        )
        try:
            raw = self._chain.invoke({"question": query, "candidates": numbered})
            scores = _parse_scores(raw, len(candidates))
        except Exception:
            logger.exception("Re-ranking call failed; falling back to fusion order")
            scores = None

        scored: list[Document] = []
        for index, doc in enumerate(candidates, start=1):
            metadata = dict(doc.metadata)
            if scores is not None and index in scores:
                metadata["relevance_score"] = scores[index]
            else:
                # fall back to the fusion-stage score so filtering/sorting still works
                metadata["relevance_score"] = metadata.get("fusion_score", 0.0)
            scored.append(Document(page_content=doc.page_content, metadata=metadata))

        if scores is None:
            logger.warning("Re-ranker returned no usable scores for %d candidate(s)", len(candidates))

        return sorted(scored, key=lambda d: d.metadata["relevance_score"], reverse=True)
