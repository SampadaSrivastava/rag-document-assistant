"""ChromaDB wrapper. All Chroma-specific code lives here, so swapping to Pinecone/Qdrant means editing one file."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.errors import VectorStoreError

logger = logging.getLogger(__name__)


def _doc_filter(doc_ids: list[str]) -> dict:
    return {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}


class DocumentStore:
    """One Chroma collection per browser session, with a doc_id on every chunk.

    The per-session collection keeps different users apart; the doc_id filter keeps
    different documents apart inside a session.
    """

    def __init__(self, session_id: str, embeddings: Embeddings, persist_dir: Path | None = None) -> None:
        try:
            self._store = Chroma(
                collection_name=f"session_{session_id}",
                embedding_function=embeddings,
                persist_directory=str(persist_dir) if persist_dir else None,
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.exception("Could not open Chroma collection")
            raise VectorStoreError("The vector database could not be opened.") from exc

    def add_chunks(self, chunks: list[Document]) -> None:
        try:
            self._store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        except Exception as exc:
            logger.exception("Chroma add failed")
            raise VectorStoreError("The document could not be saved to the vector database.") from exc

    def delete_document(self, doc_id: str) -> None:
        try:
            self._store.delete(where={"doc_id": doc_id})
        except Exception as exc:
            logger.exception("Chroma delete failed")
            raise VectorStoreError("The document could not be removed from the vector database.") from exc

    def search(self, query: str, doc_ids: list[str], k: int) -> list[tuple[Document, float]]:
        """Return (chunk, similarity) pairs; similarity = 1 - cosine distance, higher is better."""
        if not doc_ids:
            return []
        try:
            results = self._store.similarity_search_with_score(query, k=k, filter=_doc_filter(doc_ids))
        except Exception as exc:
            logger.exception("Chroma search failed")
            raise VectorStoreError("Searching the vector database failed.") from exc
        return [(doc, 1.0 - distance) for doc, distance in results]

    def sample_chunks(self, doc_ids: list[str], limit: int) -> list[Document]:
        """Pick up to `limit` chunks spread evenly through each document (used for summaries)."""
        if not doc_ids:
            return []
        try:
            raw = self._store.get(where=_doc_filter(doc_ids), include=["documents", "metadatas"])
        except Exception as exc:
            logger.exception("Chroma get failed")
            raise VectorStoreError("Reading the vector database failed.") from exc
        by_doc: dict[str, list[Document]] = {}
        for text, meta in zip(raw["documents"], raw["metadatas"]):
            by_doc.setdefault(meta["doc_id"], []).append(Document(page_content=text, metadata=meta))
        per_doc = max(1, limit // len(by_doc)) if by_doc else 0
        sampled: list[Document] = []
        for chunks in by_doc.values():
            chunks.sort(key=lambda d: d.metadata["chunk_index"])
            if len(chunks) <= per_doc:
                sampled.extend(chunks)
            else:
                step = len(chunks) / per_doc
                sampled.extend(chunks[int(i * step)] for i in range(per_doc))
        return sampled
