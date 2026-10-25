"""Retrieval: top-k similarity search with a score threshold and per-document balancing."""
from __future__ import annotations

import math
from dataclasses import dataclass

from rag.vector_store import DocumentStore


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    filename: str
    page: int
    doc_id: str
    chunk_id: str
    score: float


def retrieve(
    store: DocumentStore, query: str, doc_ids: list[str], top_k: int, score_threshold: float
) -> list[RetrievedChunk]:
    """Find the chunks most similar to `query`.

    With several documents, each is searched separately so one long document cannot crowd out the
    others (e.g. a resume vs. a job description). Results below the threshold are dropped.
    """
    if not doc_ids or not query.strip():
        return []
    per_doc = top_k if len(doc_ids) == 1 else max(2, math.ceil(top_k / len(doc_ids)))
    hits: list[RetrievedChunk] = []
    for doc_id in doc_ids:
        for doc, score in store.search(query, [doc_id], per_doc):
            if score >= score_threshold:
                meta = doc.metadata
                hits.append(RetrievedChunk(doc.page_content, meta["filename"], int(meta["page"]), meta["doc_id"], meta["chunk_id"], score))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[: max(top_k, per_doc * len(doc_ids))]
