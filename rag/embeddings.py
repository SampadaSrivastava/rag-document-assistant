"""Embedding model factory. The model name comes from settings, so it can be swapped without code changes."""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from utils.errors import EmbeddingError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_embeddings(model_name: str) -> Embeddings:
    """Load the sentence-transformers model once per process (loading takes seconds, so we cache it)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},  # unit vectors -> cosine similarity is well-behaved
        )
    except Exception as exc:
        logger.exception("Could not load embedding model %s", model_name)
        raise EmbeddingError(
            "The embedding model could not be loaded. Check your internet connection (the model downloads on first use) and EMBEDDING_MODEL."
        ) from exc
