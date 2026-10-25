"""Ingestion pipeline: PDF bytes -> text -> chunks -> embeddings -> ChromaDB."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings
from rag.text_splitter import split_pages
from rag.vector_store import DocumentStore
from utils.helpers import make_doc_id
from utils.pdf_utils import extract_pages


@dataclass(frozen=True)
class DocInfo:
    doc_id: str
    filename: str
    size_bytes: int
    page_count: int
    chunk_count: int


def ingest_pdf(store: DocumentStore, filename: str, data: bytes, settings: Settings) -> DocInfo:
    """Process one PDF. Embedding happens inside store.add_chunks (Chroma calls the embedding model)."""
    doc_id = make_doc_id(data)
    extracted = extract_pages(data)
    chunks = split_pages(extracted.pages, doc_id, filename, settings.chunk_size, settings.chunk_overlap)
    store.add_chunks(chunks)
    return DocInfo(doc_id, filename, len(data), extracted.page_count, len(chunks))
