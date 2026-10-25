import re
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding: shared words -> similar vectors. No model download needed."""

    DIM = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(token.encode()) % self.DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeLLM:
    def __init__(self, reply="Answer text."):
        self.reply, self.calls = reply, []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.reply)


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def settings():
    return Settings("fake-key", "gemini-test", "fake", None, 300, 50, 4, 0.2, 25, 5)


@pytest.fixture
def store():
    from rag.vector_store import DocumentStore
    import uuid

    return DocumentStore(uuid.uuid4().hex[:12], FakeEmbeddings(), None)
