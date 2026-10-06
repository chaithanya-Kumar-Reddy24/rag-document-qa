"""LLM building blocks: the chat model, the query rewriter and the answer generator.

Both chains use LangChain Expression Language (LCEL): ``prompt | llm | parser``.
"""

from __future__ import annotations

import logging
from typing import Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.config import Settings
from src.exceptions import LLMError, RAGError
from src.prompts import RAG_PROMPT, REWRITE_PROMPT
from src.utils import truncate

logger = logging.getLogger(__name__)


def build_llm(settings: Settings, model: str | None = None) -> BaseChatModel:
    """Create the local Ollama chat model.

    The model runs locally through the Ollama service, so no OpenAI API
    key or paid API credits are required.
    """
    return ChatOllama(
        model=model or settings.chat_model,
        temperature=settings.temperature,
    )


class QueryRewriter:
    """Turns a follow-up question into a standalone retrieval query."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = REWRITE_PROMPT | llm | StrOutputParser()

    def rewrite(self, question: str, chat_history: str) -> str:
        if not chat_history.strip():
            return question  # first question: nothing to resolve, save an API call
        try:
            rewritten = self._chain.invoke({"chat_history": chat_history, "question": question})
        except Exception:
            logger.exception("Query rewriting failed; using the original question")
            return question
        rewritten = rewritten.strip().strip('"')
        if not rewritten:
            return question
        logger.info("Query rewritten: %s", truncate(rewritten))
        return rewritten


class AnswerGenerator:
    """Generates an answer strictly from the supplied context."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._chain = RAG_PROMPT | llm | StrOutputParser()
        # Used for cost estimation (src/observability.py); not every BaseChatModel
        # exposes the same attribute name, so try the common ones and fall back.
        self.model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"

    def generate(self, question: str, context: str, callbacks: list | None = None) -> str:
        config = {"callbacks": callbacks} if callbacks else None
        try:
            return self._chain.invoke({"context": context, "question": question}, config=config).strip()
        except RAGError:
            raise
        except Exception as exc:
            logger.exception("LLM generation failed")
            raise LLMError() from exc

    def stream(self, question: str, context: str) -> Iterator[str]:
        """Yield the answer piece by piece (used when not running inside the graph)."""
        try:
            yield from self._chain.stream({"context": context, "question": question})
        except Exception as exc:
            logger.exception("LLM streaming failed")
            raise LLMError() from exc
