"""Local embedding model factory.

Uses Sentence Transformers locally for both document and query embeddings.
The same embedding model is used for indexing and searching so all vectors
remain in the same vector space.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Create the local Sentence Transformers embedding model."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )