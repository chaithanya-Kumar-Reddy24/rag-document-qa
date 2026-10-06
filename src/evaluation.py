"""RAG evaluation: measures whether retrieval and generation are actually good,
not just whether the pipeline runs.

Metrics (each 0.0-1.0, ``None`` when it couldn't be computed):

- ``retrieval_relevance`` - mean cosine similarity of the retrieved chunks. Computed
  directly from the retriever's own scores; no LLM call needed.
- ``context_precision``   - fraction of retrieved chunks an LLM judge marks as
  actually relevant to the question. Needs only a question.
- ``context_recall``      - LLM judge: does the retrieved context contain what's
  needed to produce the reference answer? Needs an ``expected_answer``.
- ``groundedness``        - LLM judge: is every claim in the generated answer
  supported by the retrieved context? Needs only a question.
- ``answer_correctness``  - LLM judge: how well does the generated answer match the
  reference answer? Needs an ``expected_answer``.

A judge score always comes back as ``None`` (never 0.0) when the judge call fails or
returns something unparseable, so a bad judge response can't silently look like "no
evidence" or "completely wrong" in the dashboard - failures are visibly excluded from
the average rather than dragging it down.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.graph import RAGPipeline, is_not_found
from src.prompts import (
    ANSWER_CORRECTNESS_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT,
    GROUNDEDNESS_PROMPT,
)

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_METRIC_NAMES = (
    "retrieval_relevance",
    "context_precision",
    "context_recall",
    "groundedness",
    "answer_correctness",
)
_METRIC_LABELS = {
    "retrieval_relevance": "Retrieval relevance",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "groundedness": "Groundedness",
    "answer_correctness": "Answer correctness",
    "avg_latency_seconds": "Avg. latency (s)",
}


@dataclass(frozen=True)
class EvalCase:
    """One row of an evaluation dataset."""

    question: str
    expected_answer: str = ""
    expected_sources: tuple[str, ...] = ()


@dataclass
class EvalResult:
    """Outcome of running one ``EvalCase`` through the pipeline and the judges."""

    question: str
    expected_answer: str
    generated_answer: str
    search_query: str
    context: str
    sources: list[str]
    retrieval_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    groundedness: float | None = None
    answer_correctness: float | None = None
    latency_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def scores(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in _METRIC_NAMES}


def _parse_json(raw: str) -> dict | None:
    cleaned = _JSON_FENCE.sub("", raw.strip()).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_eval_dataset_csv(data: str) -> list[EvalCase]:
    """Parse a CSV with columns: question, expected_answer (optional), expected_sources
    (optional, ``;``-separated filenames). Rows without a non-empty question are skipped.
    """
    reader = csv.DictReader(io.StringIO(data))
    cases: list[EvalCase] = []
    for row in reader:
        question = (row.get("question") or "").strip()
        if not question:
            continue
        expected_answer = (row.get("expected_answer") or "").strip()
        raw_sources = (row.get("expected_sources") or "").strip()
        expected_sources = tuple(s.strip() for s in raw_sources.split(";") if s.strip())
        cases.append(
            EvalCase(question=question, expected_answer=expected_answer, expected_sources=expected_sources)
        )
    return cases


def results_to_csv(results: Sequence[EvalResult]) -> str:
    """Serialize evaluation results back to CSV (for download from the UI)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "question",
            "expected_answer",
            "generated_answer",
            "sources",
            *_METRIC_NAMES,
            "latency_seconds",
        ]
    )
    for result in results:
        writer.writerow(
            [
                result.question,
                result.expected_answer,
                result.generated_answer,
                "; ".join(result.sources),
                *(getattr(result, name) for name in _METRIC_NAMES),
                round(result.latency_seconds, 3),
            ]
        )
    return buffer.getvalue()


def aggregate_scores(results: Sequence[EvalResult]) -> dict[str, float | None]:
    """Mean of each metric across results, ignoring cases where it's ``None``."""
    aggregates: dict[str, float | None] = {}
    for name in _METRIC_NAMES:
        values = [v for r in results if (v := getattr(r, name)) is not None]
        aggregates[name] = round(sum(values) / len(values), 3) if values else None
    return aggregates


def compare_retrieval_strategies(
    cases: Sequence[EvalCase],
    pipelines: dict[str, RAGPipeline],
    judge_llm: BaseChatModel,
    *,
    doc_ids: Sequence[str] | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    on_progress=None,
) -> dict[str, list[EvalResult]]:
    """Run the same evaluation dataset through several retrieval strategies.

    ``pipelines`` maps a strategy label (e.g. "Vector", "Hybrid", "Hybrid + Reranker")
    to the ``RAGPipeline`` built with that strategy's retriever. All strategies are
    judged by the *same* ``judge_llm`` instance, so scores across strategies are
    directly comparable — differences reflect the retriever, not judging drift.

    ``on_progress(strategy_index, strategy_count, case_index, case_count)`` is called
    after every case, if given, so a UI can show combined progress across strategies.
    """
    results: dict[str, list[EvalResult]] = {}
    strategy_count = len(pipelines)
    for strategy_index, (label, pipeline) in enumerate(pipelines.items(), start=1):
        evaluator = Evaluator(pipeline, judge_llm)

        def _forward(done: int, total: int, _label=label, _index=strategy_index) -> None:
            if on_progress is not None:
                on_progress(_index, strategy_count, done, total)

        results[label] = evaluator.evaluate_dataset(
            cases, doc_ids=doc_ids, top_k=top_k, threshold=threshold, on_progress=_forward
        )
    return results


def comparison_table(results_by_strategy: dict[str, list[EvalResult]]) -> list[dict]:
    """One row per metric (+ latency), one column per strategy — the shape a UI table
    or a Markdown table (``metric | Vector | Hybrid | Hybrid + Reranker``) wants."""
    rows: list[dict] = []
    for name in (*_METRIC_NAMES, "avg_latency_seconds"):
        row: dict[str, object] = {"metric": _METRIC_LABELS.get(name, name)}
        for label, results in results_by_strategy.items():
            if name == "avg_latency_seconds":
                value = round(sum(r.latency_seconds for r in results) / len(results), 3) if results else None
            else:
                value = aggregate_scores(results).get(name)
            row[label] = value
        rows.append(row)
    return rows


class Evaluator:
    """Runs an evaluation dataset through the RAG pipeline and scores it with an LLM judge."""

    def __init__(self, pipeline: RAGPipeline, judge_llm: BaseChatModel) -> None:
        self._pipeline = pipeline
        self._precision_chain = CONTEXT_PRECISION_PROMPT | judge_llm | StrOutputParser()
        self._recall_chain = CONTEXT_RECALL_PROMPT | judge_llm | StrOutputParser()
        self._groundedness_chain = GROUNDEDNESS_PROMPT | judge_llm | StrOutputParser()
        self._correctness_chain = ANSWER_CORRECTNESS_PROMPT | judge_llm | StrOutputParser()

    # ------------------------------------------------------------- judges
    def _judge_precision(self, question: str, context: str, block_count: int) -> float | None:
        if block_count == 0:
            return None
        try:
            raw = self._precision_chain.invoke({"question": question, "context": context})
            parsed = _parse_json(raw)
            relevant = parsed.get("relevant_blocks") if parsed else None
            if not isinstance(relevant, list):
                return None
            return _clamp01(len(relevant) / block_count)
        except Exception:
            logger.exception("context_precision judging failed")
            return None

    def _judge_score(self, chain, inputs: dict) -> float | None:
        try:
            raw = chain.invoke(inputs)
            parsed = _parse_json(raw)
            score = parsed.get("score") if parsed else None
            return _clamp01(float(score)) if isinstance(score, (int, float)) else None
        except Exception:
            logger.exception("judge scoring failed")
            return None

    # ------------------------------------------------------------- public
    def evaluate_case(
        self,
        case: EvalCase,
        *,
        doc_ids: Sequence[str] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> EvalResult:
        """Run one question through the pipeline, then score it with the judges that apply."""
        started = time.perf_counter()
        state = self._pipeline.ask(case.question, doc_ids=doc_ids, top_k=top_k, threshold=threshold)
        latency = time.perf_counter() - started

        context = state.get("context", "")
        answer = state.get("answer", "")
        documents = state.get("retrieved_documents", [])
        source_names = sorted({str(doc.metadata.get("source", "unknown")) for doc in documents})
        notes: list[str] = []

        relevance_scores = [
            float(doc.metadata["relevance_score"])
            for doc in documents
            if doc.metadata.get("relevance_score") is not None
        ]
        retrieval_relevance = (
            round(sum(relevance_scores) / len(relevance_scores), 3) if relevance_scores else None
        )
        if not documents:
            notes.append("No chunks retrieved.")

        context_precision = self._judge_precision(case.question, context, len(documents))

        answered = not is_not_found(answer)
        groundedness = (
            self._judge_score(self._groundedness_chain, {"context": context, "answer": answer})
            if answered and context
            else None
        )
        if not answered:
            notes.append("Pipeline returned the not-found message.")

        context_recall = None
        answer_correctness = None
        if case.expected_answer:
            context_recall = self._judge_score(
                self._recall_chain,
                {"question": case.question, "expected_answer": case.expected_answer, "context": context},
            )
            answer_correctness = self._judge_score(
                self._correctness_chain,
                {
                    "question": case.question,
                    "expected_answer": case.expected_answer,
                    "generated_answer": answer,
                },
            )
        else:
            notes.append("No expected_answer provided; context_recall and answer_correctness skipped.")

        return EvalResult(
            question=case.question,
            expected_answer=case.expected_answer,
            generated_answer=answer,
            search_query=state.get("search_query", case.question),
            context=context,
            sources=source_names,
            retrieval_relevance=retrieval_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            groundedness=groundedness,
            answer_correctness=answer_correctness,
            latency_seconds=latency,
            notes=notes,
        )

    def evaluate_dataset(
        self,
        cases: Sequence[EvalCase],
        *,
        doc_ids: Sequence[str] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        on_progress=None,
    ) -> list[EvalResult]:
        """Evaluate every case in order, calling ``on_progress(done, total)`` after each."""
        results: list[EvalResult] = []
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            results.append(self.evaluate_case(case, doc_ids=doc_ids, top_k=top_k, threshold=threshold))
            if on_progress is not None:
                on_progress(index, total)
        return results
