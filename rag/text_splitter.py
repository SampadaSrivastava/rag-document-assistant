"""Chunking. Pages are split independently so every chunk maps to exactly one page (accurate citations)."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.pdf_utils import PageText


def split_pages(
    pages: list[PageText], doc_id: str, filename: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split page text into overlapping chunks carrying filename / page / doc_id / chunk_id metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, chunk_size // 2),
        separators=["\n\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            index = len(chunks)
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page.page_number,
                        "chunk_index": index,
                        "chunk_id": f"{doc_id}-{index:04d}",
                    },
                )
            )
    return chunks
