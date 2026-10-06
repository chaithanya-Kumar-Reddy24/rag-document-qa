"""Streamlit UI for the RAG-Based AI Document Q&A System.

Run from the project root:  streamlit run app.py
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import Settings
from src.document_loader import supported_extensions
from src.embeddings import get_embeddings
from src.evaluation import (
    EvalCase,
    Evaluator,
    aggregate_scores,
    comparison_table,
    compare_retrieval_strategies,
    load_eval_dataset_csv,
    results_to_csv,
)
from src.exceptions import RAGError, to_user_message
from src.graph import RAGPipeline
from src.ingestion import IngestionResult, IngestionService
from src.keyword_search import BM25Index, build_bm25_index
from src.memory import ConversationMemory
from src.rag_chain import AnswerGenerator, QueryRewriter, build_llm
from src.reranker import LLMReranker
from src.retriever import DocumentRetriever, HybridRetriever
from src.utils import setup_logging, truncate
from src.vector_store import IndexedDocument, VectorStoreManager

logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")
# =========================
# 🎨 CUSTOM UI ANIMATIONS & STYLING
# =========================

st.markdown(
    """
    <style>
    .main { animation: pageFadeIn 0.75s ease-out; }

    @keyframes pageFadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animated-title {
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 850;
        line-height: 1.15;
        text-align: center;
        margin: 8px 0;
        background: linear-gradient(90deg, #7c3aed, #2563eb, #06b6d4, #7c3aed);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientMove 6s ease infinite;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .animated-subtitle {
        text-align: center;
        font-size: 1.05rem;
        opacity: 0.78;
        margin-bottom: 20px;
        animation: subtitleFade 1s ease-out;
    }

    @keyframes subtitleFade {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 0.78; transform: translateY(0); }
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 750;
        margin: 22px 0 12px;
        animation: slideIn 0.6s ease-out;
    }

    .section-subtitle {
        font-size: 0.92rem;
        opacity: 0.72;
        margin-top: -5px;
        margin-bottom: 14px;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-18px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .glass-card {
        padding: 18px 20px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(124,58,237,.10), rgba(37,99,235,.06));
        border: 1px solid rgba(124,58,237,.18);
        box-shadow: 0 8px 28px rgba(0,0,0,.08);
        transition: transform .3s ease, box-shadow .3s ease;
        animation: cardAppear .7s ease-out;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 34px rgba(99,102,241,.20);
    }

    @keyframes cardAppear {
        from { opacity: 0; transform: translateY(18px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="stMetric"] {
        padding: 14px;
        border-radius: 16px;
        border: 1px solid rgba(124,58,237,.14);
        background: rgba(255,255,255,.035);
        transition: transform .25s ease, box-shadow .25s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(99,102,241,.18);
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 650;
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 8px 20px rgba(99,102,241,.28);
    }

    .stButton > button:active, .stDownloadButton > button:active {
        transform: scale(.97);
    }

    [data-testid="stFileUploader"] {
        border-radius: 18px;
        padding: 8px;
        border: 1px solid rgba(124,58,237,.10);
        transition: transform .3s ease, box-shadow .3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(99,102,241,.18);
    }

    [data-testid="stChatMessage"] {
        animation: chatAppear .45s ease-out;
    }

    @keyframes chatAppear {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="stChatInput"] {
        transition: box-shadow .3s ease, transform .3s ease;
    }

    [data-testid="stChatInput"]:focus-within {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99,102,241,.18);
    }

    [data-testid="stExpander"] {
        border-radius: 14px;
        transition: box-shadow .3s ease, transform .3s ease;
    }

    [data-testid="stExpander"]:hover {
        box-shadow: 0 6px 20px rgba(99,102,241,.13);
    }

    [data-testid="stAlert"] {
        animation: alertAppear .45s ease-out;
    }

    @keyframes alertAppear {
        from { opacity: 0; transform: scale(.98); }
        to { opacity: 1; transform: scale(1); }
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        animation: cardAppear .5s ease-out;
    }

    [data-testid="stProgress"] {
        animation: progressGlow 1.8s ease-in-out infinite;
    }

    @keyframes progressGlow {
        0%, 100% { opacity: .82; }
        50% { opacity: 1; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Cached resources (created once per server process, not on every rerun)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_vector_store(persist_dir: str, embedding_model: str) -> VectorStoreManager:
    settings = replace(
        Settings.from_env(),
        chroma_persist_directory=Path(persist_dir),
        embedding_model=embedding_model,
    )
    return VectorStoreManager(settings, get_embeddings(settings))


@st.cache_resource(show_spinner=False)
def get_keyword_index(persist_dir: str, embedding_model: str, store_version: int) -> BM25Index:
    """Rebuilt whenever ``store_version`` changes (i.e. after any index/delete/clear)."""
    store = get_vector_store(persist_dir, embedding_model)
    return build_bm25_index(store)


RETRIEVAL_MODES = ("vector", "hybrid_no_rerank", "hybrid_rerank")
RETRIEVAL_MODE_LABELS = {
    "vector": "Vector",
    "hybrid_no_rerank": "Hybrid",
    "hybrid_rerank": "Hybrid + Reranker",
}


def _mode_for_sidebar(hybrid_enabled: bool) -> str:
    return "hybrid_no_rerank" if hybrid_enabled else "vector"


@st.cache_resource(show_spinner=False)
def get_pipeline(
    chat_model: str,
    persist_dir: str,
    embedding_model: str,
    retrieval_mode: str,
    candidate_pool_size: int,
    store_version: int,
) -> RAGPipeline:
    """Build a pipeline for one retrieval strategy — "vector" (cosine similarity
    only), "hybrid_no_rerank" (vector + BM25, RRF-fused, no LLM re-ranking), or
    "hybrid_rerank" (the same, plus LLM re-ranking). Same three strategies the
    Evaluation Dashboard's A/B comparison runs side by side."""
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode!r}")

    settings = Settings.from_env()
    store = get_vector_store(persist_dir, embedding_model)
    llm = build_llm(settings, chat_model)

    if retrieval_mode == "vector":
        retriever: DocumentRetriever | HybridRetriever = DocumentRetriever(
            store, settings.top_k, settings.relevance_threshold
        )
    else:
        keyword_index = get_keyword_index(persist_dir, embedding_model, store_version)
        enable_reranking = retrieval_mode == "hybrid_rerank"
        retriever = HybridRetriever(
            store,
            LLMReranker(llm) if enable_reranking else None,
            keyword_index_provider=lambda: keyword_index,
            default_top_k=settings.top_k,
            default_threshold=settings.relevance_threshold,
            candidate_pool_size=candidate_pool_size,
            rrf_k=settings.rrf_k,
            enable_reranking=enable_reranking,
        )

    return RAGPipeline(retriever, QueryRewriter(llm), AnswerGenerator(llm), settings.max_context_chars)


@st.cache_resource(show_spinner=False)
def get_evaluator(
    chat_model: str,
    persist_dir: str,
    embedding_model: str,
    retrieval_mode: str,
    candidate_pool_size: int,
    store_version: int,
) -> Evaluator:
    """A separate LLM instance acts as the judge, so its calls never mix with the
    query-rewriting / answer-generation / re-ranking LLM calls made inside the pipeline."""
    settings = Settings.from_env()
    pipeline = get_pipeline(
        chat_model, persist_dir, embedding_model, retrieval_mode, candidate_pool_size, store_version
    )
    judge_llm = build_llm(settings, chat_model)
    return Evaluator(pipeline, judge_llm)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SidebarConfig:
    uploaded_files: list[Any]
    chunk_size: int
    chunk_overlap: int
    top_k: int
    threshold: float
    chat_model: str
    hybrid_enabled: bool
    candidate_pool_size: int
    index_clicked: bool
    clear_clicked: bool


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def render_sidebar(settings: Settings) -> SidebarConfig:
    with st.sidebar:
        st.markdown(
    """
    <div class="section-title">
        📁 Documents
    </div>
    """,
    unsafe_allow_html=True,
)
        extensions = [ext.lstrip(".") for ext in supported_extensions()]
        uploaded = st.file_uploader(
            "📤 Upload Your Documents", type=extensions, accept_multiple_files=True
        )
        st.caption("Supported file types: " + ", ".join(e.upper() for e in extensions))

        st.header("⚙️ Settings")
        chunk_size = st.slider(
            "Chunk size (characters)", 200, 4000, int(_clamp(settings.chunk_size, 200, 4000)), step=100
        )
        chunk_overlap = st.slider(
            "Chunk overlap (characters)", 0, 1000, int(_clamp(settings.chunk_overlap, 0, 1000)), step=50
        )
        top_k = st.slider("Top-K retrieved chunks", 1, 20, int(_clamp(settings.top_k, 1, 20)))
        threshold = st.slider(
            "Relevance threshold",
            0.0,
            1.0,
            float(_clamp(settings.relevance_threshold, 0.0, 1.0)),
            step=0.05,
            help="Minimum relevance score for a chunk to be sent to the LLM. "
            "Lower it if you often get 'could not find the answer'.",
        )
        options = [settings.chat_model]
        chat_model = st.selectbox("Chat model", options, index=0)
        st.caption(
            f"Embedding model: `{settings.embedding_model}`. Chunk size and overlap apply "
            "to documents indexed after you change them."
        )

        st.subheader("🔎 Hybrid search")
        hybrid_enabled = st.toggle(
            "Vector + keyword (BM25) hybrid search",
            value=settings.enable_hybrid_search,
            help="On: combine semantic (vector) search with exact-term (BM25) search "
            "via Reciprocal Rank Fusion, then apply Top-K and the relevance threshold. "
            "This fast mode does not make an extra LLM re-ranking call. Off: vector search only.",
        )
        candidate_pool_size = st.slider(
            "Candidate pool size",
            top_k,
            50,
            int(_clamp(settings.candidate_pool_size, top_k, 50)),
            disabled=not hybrid_enabled,
            help="How many chunks vector search and BM25 each contribute before "
            "fusion narrows them down to Top-K.",
        )

        overlap_ok = chunk_overlap < chunk_size
        if not overlap_ok:
            st.error("Chunk overlap must be smaller than chunk size.")
        index_clicked = st.button(
            "Index documents", type="primary", disabled=not uploaded or not overlap_ok
        )

        st.divider()
        st.subheader("🗑️ Vector database")
        confirm = st.checkbox("Yes, delete everything that is indexed")
        clear_clicked = st.button("Clear vector database", disabled=not confirm)

    return SidebarConfig(
        uploaded_files=list(uploaded or []),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        threshold=threshold,
        chat_model=chat_model,
        hybrid_enabled=hybrid_enabled,
        candidate_pool_size=candidate_pool_size,
        index_clicked=index_clicked,
        clear_clicked=clear_clicked,
    )


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def index_files(service: IngestionService, config: SidebarConfig) -> list[IngestionResult]:
    results: list[IngestionResult] = []
    total = len(config.uploaded_files)
    progress = st.progress(0.0, text="Starting...")
    for number, uploaded in enumerate(config.uploaded_files, start=1):
        progress.progress((number - 1) / total, text=f"Indexing {uploaded.name} ({number}/{total})")
        results.append(
            service.ingest(uploaded.name, uploaded.getvalue(), config.chunk_size, config.chunk_overlap)
        )
    progress.empty()
    return results


def render_documents_table(documents: list[IndexedDocument], last_run: list[IngestionResult]) -> None:
    rows = [
        {
            "Filename": doc.filename,
            "File type": doc.file_type.upper(),
            "Pages": str(doc.pages) if doc.pages else "-",
            "Chunks": str(doc.chunks),
            "Indexing status": "✅ Indexed",
        }
        for doc in documents
    ]
    for result in last_run:
        if result.status == "indexed":
            continue  # already listed above from the database
        status = "⏭️ Skipped (duplicate)" if result.status == "duplicate" else f"❌ Failed: {result.message}"
        rows.append(
            {
                "Filename": result.filename,
                "File type": Path(result.filename).suffix.lstrip(".").upper() or "-",
                "Pages": "-",
                "Chunks": "-",
                "Indexing status": status,
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.info("No documents yet. Upload files in the sidebar and click **Index documents**.")


def make_labels(documents: list[IndexedDocument]) -> dict[str, str]:
    """Map a display label to a doc_id (adds a short id when filenames repeat)."""
    counts = Counter(doc.filename for doc in documents)
    return {
        (doc.filename if counts[doc.filename] == 1 else f"{doc.filename} ({doc.doc_id[:6]})"): doc.doc_id
        for doc in documents
    }


# --------------------------------------------------------------------------- #
# Chat rendering
# --------------------------------------------------------------------------- #
def _format_ms(value: Any) -> str:
    return f"{value:.0f} ms" if isinstance(value, (int, float)) else "—"


def render_performance_details(metrics: dict[str, Any]) -> None:
    """Per-query latency breakdown, chunk counts, and token/cost estimate."""
    if not metrics:
        return
    with st.expander(f"⏱ Performance ({_format_ms(metrics.get('total_ms'))} total)"):
        stage_cols = st.columns(5)
        stage_cols[0].metric("Query rewrite", _format_ms(metrics.get("rewrite_ms")))
        stage_cols[1].metric("Vector search", _format_ms(metrics.get("vector_ms")))
        stage_cols[2].metric("Keyword search", _format_ms(metrics.get("keyword_ms")))
        stage_cols[3].metric("Re-ranking", _format_ms(metrics.get("rerank_ms")))
        stage_cols[4].metric("LLM generation", _format_ms(metrics.get("generation_ms")))

        info_cols = st.columns(4)
        info_cols[0].metric("Chunks retrieved", metrics.get("retrieved_chunks", "—"))
        info_cols[1].metric("Chunks used", metrics.get("final_chunks", "—"))
        avg_score = metrics.get("avg_retrieval_score")
        info_cols[2].metric("Avg. relevance", f"{avg_score:.2f}" if avg_score is not None else "—")
        total_tokens = metrics.get("total_tokens")
        info_cols[3].metric("Tokens used", total_tokens if total_tokens else "—")

        cost = metrics.get("estimated_cost_usd")
        if cost is not None:
            st.caption(
                f"Estimated cost: ${cost:.5f} (illustrative pricing table, not a live "
                f"feed — see README 'Observability' section)."
            )
        elif total_tokens:
            st.caption("Estimated cost: — (no price on file for this model).")


def render_assistant_details(message: dict[str, Any]) -> None:
    """Show sources and the exact retrieved chunks under an answer."""
    sources = message.get("sources") or []
    chunks = message.get("chunks") or []

    if sources:
        st.markdown("**Sources**")
        for source in sources:
            lines = [f"**Document:** `{source['filename']}`"]
            if source.get("location"):
                kind, _, value = source["location"].partition(" ")
                lines.append(f"**{kind}:** {value}")
            st.markdown("  \n".join(lines))

    if chunks:
        st.markdown(f"**Retrieved text** ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})")
        for number, chunk in enumerate(chunks, start=1):
            where = f" · {chunk['location']}" if chunk["location"] else ""
            flag = " · ⚠️ flagged" if chunk.get("flagged") else ""
            title = f"[{number}] {chunk['filename']}{where} · relevance {chunk['score']:.2f}{flag}"
            with st.expander(title):
                if chunk.get("flagged"):
                    categories = ", ".join(chunk.get("flag_categories", [])) or "instruction-like phrasing"
                    st.warning(
                        f"This passage contains phrasing resembling an injection attempt "
                        f"({categories}). It was treated as document content, not an "
                        f"instruction — see the README's 'Prompt-Injection Protection' section.",
                        icon="⚠️",
                    )
                st.text(chunk["text"])

    render_performance_details(message.get("metrics") or {})

    query = message.get("search_query")
    if query and query != message.get("question"):
        st.caption(f"🔎 Search query used: {query}")


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_assistant_details(message)


def answer_question(
    question: str,
    pipeline: RAGPipeline,
    doc_ids: list[str],
    config: SidebarConfig,
) -> None:
    """Run the RAG graph, stream the answer and store the result in session state."""
    memory: ConversationMemory = st.session_state.memory
    history = memory.format_history()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Searching your documents..._")
        streamed = ""
        final: dict[str, Any] | None = None
        try:
            for event in pipeline.stream(
                question,
                chat_history=history,
                doc_ids=doc_ids,
                top_k=config.top_k,
                threshold=config.threshold,
            ):
                if event.kind == "token":
                    streamed += event.token
                    placeholder.markdown(streamed + "▌")
                else:
                    final = event.state
        except Exception as exc:  # never show stack traces to users
            logger.exception("Question failed: %s", truncate(question))
            placeholder.empty()
            error_text = f"⚠️ {to_user_message(exc)}"
            st.error(to_user_message(exc))
            st.session_state.messages.append(
                {"role": "assistant", "content": error_text, "sources": [], "chunks": []}
            )
            return

        state = final or {}
        answer = state.get("answer") or streamed
        placeholder.markdown(answer)

        message = {
            "role": "assistant",
            "content": answer,
            "question": question,
            "search_query": state.get("search_query"),
            "sources": state.get("sources", []),
            "metrics": state.get("metrics", {}),
            "chunks": [
                {
                    "filename": doc.metadata.get("source", "unknown"),
                    "location": _location(doc.metadata),
                    "score": float(doc.metadata.get("relevance_score", 0.0)),
                    "text": doc.page_content,
                    "flagged": bool(doc.metadata.get("injection_flagged")),
                    "flag_categories": doc.metadata.get("injection_categories", []),
                }
                for doc in state.get("retrieved_documents", [])
            ],
        }
        render_assistant_details(message)

    st.session_state.messages.append(message)
    st.session_state.last_retrieved = len(message["chunks"])
    if state.get("metrics"):
        st.session_state.query_metrics.append(state["metrics"])
    memory.add_user_message(question)
    memory.add_assistant_message(answer)


def _location(metadata: dict[str, Any]) -> str:
    from src.utils import describe_location

    return describe_location(metadata)


# --------------------------------------------------------------------------- #
# Evaluation dashboard
# --------------------------------------------------------------------------- #
SAMPLE_EVAL_CSV = (
    "question,expected_answer,expected_sources\n"
    "What is this document about?,,\n"
    "Summarize the key points.,,\n"
)

_METRIC_LABELS = {
    "retrieval_relevance": "Retrieval relevance",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "groundedness": "Groundedness",
    "answer_correctness": "Answer correctness",
}


def init_eval_state() -> None:
    st.session_state.setdefault("eval_cases", [])
    st.session_state.setdefault("eval_results", [])
    st.session_state.setdefault("comparison_results", {})


def render_eval_dataset_editor() -> list[EvalCase]:
    st.caption(
        "One row per test question. `expected_answer` and `expected_sources` (`;`-separated "
        "filenames) are optional — leaving them blank still scores retrieval relevance, "
        "context precision, and groundedness; filling them in also scores context recall "
        "and answer correctness."
    )
    uploaded_csv = st.file_uploader("Upload a dataset CSV", type=["csv"], key="eval_csv_uploader")
    if uploaded_csv is not None:
        try:
            st.session_state.eval_cases = load_eval_dataset_csv(uploaded_csv.getvalue().decode("utf-8"))
        except Exception:
            st.error("Could not read that CSV. Expected columns: question, expected_answer, expected_sources.")

    if not st.session_state.eval_cases:
        st.session_state.eval_cases = load_eval_dataset_csv(SAMPLE_EVAL_CSV)

    rows = [
        {
            "question": case.question,
            "expected_answer": case.expected_answer,
            "expected_sources": "; ".join(case.expected_sources),
        }
        for case in st.session_state.eval_cases
    ]
    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        use_container_width=True,
        key="eval_data_editor",
        column_config={
            "question": st.column_config.TextColumn("Question", width="large"),
            "expected_answer": st.column_config.TextColumn("Expected answer (optional)", width="large"),
            "expected_sources": st.column_config.TextColumn("Expected sources (optional)"),
        },
    )
    cases = [
        EvalCase(
            question=(row.get("question") or "").strip(),
            expected_answer=(row.get("expected_answer") or "").strip(),
            expected_sources=tuple(
                s.strip() for s in (row.get("expected_sources") or "").split(";") if s.strip()
            ),
        )
        for row in edited
        if (row.get("question") or "").strip()
    ]
    st.session_state.eval_cases = cases
    return cases


def render_eval_results(results: list) -> None:
    if not results:
        return

    st.subheader("Aggregate scores")
    aggregates = aggregate_scores(results)
    columns = st.columns(len(_METRIC_LABELS))
    for column, (key, label) in zip(columns, _METRIC_LABELS.items()):
        value = aggregates.get(key)
        column.metric(label, f"{value:.2f}" if value is not None else "—")

    avg_latency = sum(r.latency_seconds for r in results) / len(results)
    st.caption(f"Average response latency: {avg_latency:.2f}s over {len(results)} question(s).")

    st.subheader("Per-question results")
    table_rows = [
        {
            "Question": r.question,
            "Answer": truncate(r.generated_answer, 120),
            "Sources": ", ".join(r.sources) or "-",
            **{label: (f"{getattr(r, key):.2f}" if getattr(r, key) is not None else "—") for key, label in _METRIC_LABELS.items()},
        }
        for r in results
    ]
    st.dataframe(table_rows, hide_index=True, use_container_width=True)

    for number, result in enumerate(results, start=1):
        with st.expander(f"[{number}] {result.question}"):
            st.markdown(f"**Generated answer:** {result.generated_answer}")
            if result.expected_answer:
                st.markdown(f"**Expected answer:** {result.expected_answer}")
            st.markdown(f"**Search query used:** {result.search_query}")
            st.markdown(f"**Sources:** {', '.join(result.sources) or 'none'}")
            if result.notes:
                st.caption(" · ".join(result.notes))
            st.text(result.context or "(no context retrieved)")

    st.download_button(
        "Download results as CSV",
        data=results_to_csv(results),
        file_name="rag_evaluation_results.csv",
        mime="text/csv",
    )


def render_comparison_results(results_by_strategy: dict[str, list]) -> None:
    if not results_by_strategy:
        return

    st.subheader("Vector vs. Hybrid vs. Hybrid + Reranker")
    st.caption(
        "Same questions, same judge, three retrieval strategies — the numbers below "
        "aren't assumed, they come straight out of this run."
    )
    table = comparison_table(results_by_strategy)
    display_rows = [
        {
            "Metric": row["metric"],
            **{
                label: (f"{value:.2f}" if isinstance(value, float) else "—")
                for label, value in row.items()
                if label != "metric"
            },
        }
        for row in table
    ]
    st.dataframe(display_rows, hide_index=True, use_container_width=True)

    combined_csv_rows = []
    for label, results in results_by_strategy.items():
        for result in results:
            combined_csv_rows.append(f"# {label}")
            combined_csv_rows.append(results_to_csv([result]).strip())
    st.download_button(
        "Download comparison results as CSV",
        data="\n".join(combined_csv_rows),
        file_name="rag_strategy_comparison.csv",
        mime="text/csv",
    )


def render_evaluation_dashboard(
    settings: Settings,
    config: SidebarConfig,
    selected_ids: list[str],
    has_documents: bool,
    store_version: int,
) -> None:
    st.markdown('<div class="section-title">🧪 Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.caption(
        "Runs each question through the same RAG pipeline used in Chat, then scores retrieval "
        "and generation quality with an LLM judge — so you can answer 'did I actually check "
        "whether this works?' with numbers, not just a demo."
    )
    init_eval_state()

    if not has_documents:
        st.info("Index at least one document before running an evaluation.")
        return

    cases = render_eval_dataset_editor()

    column_run, column_compare = st.columns(2)
    run_clicked = column_run.button("Run evaluation", type="primary", disabled=not cases)
    compare_clicked = column_compare.button(
        "Compare retrieval strategies (Vector vs. Hybrid vs. Hybrid + Reranker)",
        disabled=not cases,
        help="Runs the same question set through all three retrieval strategies and "
        "scores each with the same judge, so the differences are measured, not assumed.",
    )

    if run_clicked:
        evaluator = get_evaluator(
            config.chat_model,
            str(settings.chroma_persist_directory),
            settings.embedding_model,
            _mode_for_sidebar(config.hybrid_enabled),
            config.candidate_pool_size,
            store_version,
        )
        progress = st.progress(0.0, text="Starting evaluation...")

        def on_progress(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Evaluating question {done}/{total}...")

        try:
            st.session_state.eval_results = evaluator.evaluate_dataset(
                cases,
                doc_ids=selected_ids or None,
                top_k=config.top_k,
                threshold=config.threshold,
                on_progress=on_progress,
            )
        except RAGError as exc:
            st.error(exc.user_message)
        except Exception as exc:
            logger.exception("Evaluation run failed")
            st.error(to_user_message(exc))
        finally:
            progress.empty()

    if compare_clicked:
        settings_current = Settings.from_env()
        judge_llm = build_llm(settings_current, config.chat_model)
        pipelines = {
            RETRIEVAL_MODE_LABELS[mode]: get_pipeline(
                config.chat_model,
                str(settings.chroma_persist_directory),
                settings.embedding_model,
                mode,
                config.candidate_pool_size,
                store_version,
            )
            for mode in RETRIEVAL_MODES
        }
        progress = st.progress(0.0, text="Starting comparison...")

        def on_compare_progress(
            strategy_index: int, strategy_count: int, done: int, total: int
        ) -> None:
            overall = ((strategy_index - 1) * total + done) / (strategy_count * total)
            label = list(pipelines)[strategy_index - 1]
            progress.progress(
                overall, text=f"[{strategy_index}/{strategy_count}] {label}: question {done}/{total}"
            )

        try:
            st.session_state.comparison_results = compare_retrieval_strategies(
                cases,
                pipelines,
                judge_llm,
                doc_ids=selected_ids or None,
                top_k=config.top_k,
                threshold=config.threshold,
                on_progress=on_compare_progress,
            )
        except RAGError as exc:
            st.error(exc.user_message)
        except Exception as exc:
            logger.exception("Strategy comparison failed")
            st.error(to_user_message(exc))
        finally:
            progress.empty()

    render_eval_results(st.session_state.eval_results)
    render_comparison_results(st.session_state.comparison_results)


# --------------------------------------------------------------------------- #
# Observability dashboard
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def render_observability_dashboard() -> None:
    st.markdown('<div class="section-title">📊 Observability</div>', unsafe_allow_html=True)
    metrics_log: list[dict[str, Any]] = st.session_state.query_metrics

    if not metrics_log:
        st.caption("Ask a question in Chat to start collecting latency and token stats.")
        return

    st.caption(
        "Every number below comes from this session's actual queries — nothing here is "
        "estimated ahead of time."
    )

    total_ms = [m["total_ms"] for m in metrics_log if m.get("total_ms") is not None]
    retrieval_ms = [m["retrieval_ms"] for m in metrics_log if m.get("retrieval_ms") is not None]
    generation_ms = [m["generation_ms"] for m in metrics_log if m.get("generation_ms") is not None]
    no_context_count = sum(1 for m in metrics_log if m.get("no_context"))
    total_tokens = sum(m.get("total_tokens", 0) or 0 for m in metrics_log)
    costs = [m["estimated_cost_usd"] for m in metrics_log if m.get("estimated_cost_usd") is not None]

    row1 = st.columns(4)
    row1[0].metric("Queries this session", len(metrics_log))
    row1[1].metric("Avg. total latency", f"{_mean(total_ms):.0f} ms" if _mean(total_ms) else "—")
    row1[2].metric("Avg. retrieval latency", f"{_mean(retrieval_ms):.0f} ms" if _mean(retrieval_ms) else "—")
    row1[3].metric("Avg. LLM latency", f"{_mean(generation_ms):.0f} ms" if _mean(generation_ms) else "—")

    row2 = st.columns(4)
    row2[0].metric("Total tokens used", total_tokens or "—")
    row2[1].metric(
        "Estimated total cost",
        f"${sum(costs):.4f}" if costs else "—",
        help="Sum of per-query estimates from an illustrative pricing table — see README.",
    )
    row2[2].metric("'No context' responses", no_context_count)
    avg_chunks = _mean([m["final_chunks"] for m in metrics_log if m.get("final_chunks") is not None])
    row2[3].metric("Avg. chunks used", avg_chunks if avg_chunks is not None else "—")

    stage_labels = {
        "rewrite_ms": "Query rewrite",
        "vector_ms": "Vector search",
        "keyword_ms": "Keyword search",
        "rerank_ms": "Re-ranking",
        "generation_ms": "LLM generation",
    }
    stage_rows = []
    for key, label in stage_labels.items():
        values = [m[key] for m in metrics_log if m.get(key) is not None]
        if values:
            stage_rows.append({"stage": label, "avg_ms": _mean(values)})
    if stage_rows:
        st.caption("Average latency by stage (only stages that ran at least once are shown):")
        st.bar_chart(stage_rows, x="stage", y="avg_ms")

    with st.expander(f"Per-query log ({len(metrics_log)} quer{'y' if len(metrics_log) == 1 else 'ies'})"):
        table_rows = [
            {
                "Total (ms)": m.get("total_ms", "—"),
                "Retrieval (ms)": m.get("retrieval_ms", "—"),
                "LLM (ms)": m.get("generation_ms", "—"),
                "Chunks used": m.get("final_chunks", "—"),
                "Tokens": m.get("total_tokens", "—"),
                "Cost ($)": f"{m['estimated_cost_usd']:.5f}" if m.get("estimated_cost_usd") is not None else "—",
                "No context": "yes" if m.get("no_context") else "",
            }
            for m in metrics_log
        ]
        st.dataframe(table_rows, hide_index=True, use_container_width=True)

    if st.button("Clear observability log"):
        st.session_state.query_metrics = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def init_session_state(settings: Settings) -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_run", [])
    st.session_state.setdefault("last_retrieved", 0)
    st.session_state.setdefault("memory", ConversationMemory(settings.max_history_turns))
    st.session_state.setdefault("query_metrics", [])


def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_file)
    init_session_state(settings)

    st.markdown(
    """
    <div class="animated-title">
        📚 RAG-Based AI Document Q&A System
    </div>

    <div class="animated-subtitle">
        Upload your documents and ask questions using AI-powered retrieval.
    </div>
    """,
    unsafe_allow_html=True,
)
    st.markdown('<div class="section-subtitle">🔎 Grounded answers • 📚 Source references • ⚡ Hybrid retrieval</div>', unsafe_allow_html=True)

    config = render_sidebar(settings)

    store = None
    pipeline = None

    try:
        store = get_vector_store(
            str(settings.chroma_persist_directory),
            settings.embedding_model,
        )

        pipeline = get_pipeline(
            config.chat_model,
            str(settings.chroma_persist_directory),
            settings.embedding_model,
            _mode_for_sidebar(config.hybrid_enabled),
            config.candidate_pool_size,
            store.version,
        )

    except RAGError as exc:
        st.error(exc.user_message)
        st.stop()

    if store is None or pipeline is None:
        st.error("Could not initialize the RAG application.")
        st.stop()

    service = IngestionService(settings, store)

    if config.clear_clicked:
        try:
            store.clear()
            service.clear_uploads()
            st.session_state.last_run = []
            st.session_state.last_retrieved = 0
            st.sidebar.success("Vector database cleared.")
        except RAGError as exc:
            st.sidebar.error(exc.user_message)

    if config.index_clicked:
        with st.spinner("Extracting, chunking, embedding and indexing..."):
            st.session_state.last_run = index_files(service, config)
    try:
        documents = store.list_documents()
        total_chunks = store.count_chunks()
    except RAGError as exc:
        st.error(exc.user_message)
        st.stop()

    labels = make_labels(documents)

    # ------------------------------------------------------------ documents
    st.markdown('<div class="section-title">📄 Uploaded Documents</div>', unsafe_allow_html=True)
    render_documents_table(documents, st.session_state.last_run)

    if documents:
        with st.expander("Remove a single document"):
            choice = st.selectbox("Document to remove", list(labels))
            if st.button("Remove document"):
                doc_id = labels[choice]
                try:
                    store.delete_document(doc_id)
                    service.remove_upload(doc_id)
                    st.rerun()
                except RAGError as exc:
                    st.error(exc.user_message)

    selected_labels = st.multiselect(
        "Search in these documents",
        options=list(labels),
        default=list(labels),
        help="Only the selected documents are searched (metadata filtering).",
        disabled=not labels,
    )
    selected_ids = [labels[label] for label in selected_labels]

    columns = st.columns(4)
    columns[0].metric("Documents indexed", len(documents))
    columns[1].metric("Chunks created", total_chunks)
    columns[2].metric("Retrieved chunks (last answer)", st.session_state.last_retrieved)
    columns[3].metric("Selected documents", f"{len(selected_ids)} / {len(documents)}")

    # ----------------------------------------------------------------- chat
    st.markdown('<div class="section-title">💬 Chat With Your Documents</div>', unsafe_allow_html=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.last_retrieved = 0
        st.rerun()

    render_history()

    can_chat = bool(documents) and bool(selected_ids)
    if not documents:
        st.info("Index at least one document to start chatting.")
    elif not selected_ids:
        st.warning("Select at least one document to search.")

    question = st.chat_input("Ask a question about your documents...", disabled=not can_chat)
    if question and question.strip():
        answer_question(question.strip(), pipeline, selected_ids, config)
        st.rerun()

    # --------------------------------------------------------- evaluation
    st.divider()
    render_evaluation_dashboard(
        settings, config, selected_ids, has_documents=bool(documents), store_version=store.version
    )

    # ------------------------------------------------------ observability
    st.divider()
    render_observability_dashboard()


if __name__ == "__main__":
    main()
