"""Headless UI tests using Streamlit's AppTest. File upload dialogs aren't supported by AppTest,
so documents are ingested through the same pipeline and seeded into session state."""
import uuid
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from components import state
from config.settings import get_settings
from rag.ingest import ingest_pdf
from rag.vector_store import DocumentStore
from tests.conftest import FakeEmbeddings, FakeLLM, make_pdf

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _new_app(monkeypatch, api_key="fake-key"):
    monkeypatch.setenv("GOOGLE_API_KEY", api_key)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "")
    monkeypatch.setattr(state, "get_embeddings", lambda name: FakeEmbeddings())
    return AppTest.from_file(APP, default_timeout=30)


def _seed(at, *named_pdfs):
    session_id = uuid.uuid4().hex[:12]
    store = DocumentStore(session_id, FakeEmbeddings(), None)
    docs = {}
    for name, pages in named_pdfs:
        info = ingest_pdf(store, name, make_pdf(pages), get_settings())
        docs[info.doc_id] = info
    at.session_state["session_id"] = session_id
    at.session_state["store"] = store
    at.session_state["docs"] = docs
    at.session_state["scope"] = list(docs)
    at.session_state["sync_scope"] = True


def test_landing_page_renders_empty_state(monkeypatch):
    at = _new_app(monkeypatch).run()
    assert not at.exception
    assert "Upload documents" in [b.label for b in at.button]
    assert "No documents yet" in " ".join(m.value for m in at.markdown)


def test_missing_api_key_warning(monkeypatch):
    at = _new_app(monkeypatch, api_key="").run()
    assert not at.exception and any("GOOGLE_API_KEY" in w.value for w in at.sidebar.warning)


def test_chat_flow_with_sources_and_clear(monkeypatch):
    monkeypatch.setattr("rag.qa_chain.get_llm", lambda key, model: FakeLLM("You know Python and SQL."))
    at = _new_app(monkeypatch)
    _seed(at, ("resume.pdf", ["Skills: Python, SQL, Streamlit.", "Education: B.Tech."]))
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("Which Python skills are listed?").run()
    assert not at.exception
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["sources"] and "Python" in messages[1]["content"]
    assert any("resume.pdf — Page 1" in m.value for m in at.markdown)
    next(b for b in at.sidebar.button if b.label == "Clear conversation").click().run()
    assert at.session_state["messages"] == []


def test_invalid_api_key_shows_friendly_error(monkeypatch):
    class Boom:
        def invoke(self, messages):
            raise RuntimeError("400 API key not valid. Please pass a valid API key.")

    monkeypatch.setattr("rag.qa_chain.get_llm", lambda key, model: Boom())
    at = _new_app(monkeypatch)
    _seed(at, ("resume.pdf", ["Skills: Python, SQL."]))
    at.run()
    at.chat_input[0].set_value("Which Python skills?").run()
    assert not at.exception
    assert any("API key was rejected" in e.value for e in at.error)


def test_multi_document_chat_and_summary_and_remove(monkeypatch):
    monkeypatch.setattr("rag.qa_chain.get_llm", lambda key, model: FakeLLM("Summary text"))
    at = _new_app(monkeypatch)
    _seed(at, ("resume.pdf", ["Skills: Python, SQL."]), ("job.pdf", ["Requirements: Kubernetes and Docker."]))
    at.run()
    at.chat_input[0].set_value("skills and requirements").run()
    sources = at.session_state["messages"][1]["sources"]
    assert {s.filename for s in sources} == {"resume.pdf", "job.pdf"}
    at.session_state["view"] = "Summary"
    at.run()
    next(b for b in at.button if b.label == "Key points").click().run()
    assert not at.exception and any("Summary text" in m.value for m in at.markdown)
    at.session_state["view"] = "Documents"
    at.run()
    next(b for b in at.button if b.key and b.key.startswith("rm_")).click().run()
    assert not at.exception and len(at.session_state["docs"]) == 1
