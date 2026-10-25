import pytest

from rag.ingest import ingest_pdf
from rag.prompts import NOT_FOUND_MESSAGE
from rag.qa_chain import answer_question, summarize
from rag.retriever import retrieve
from rag.text_splitter import split_pages
from utils.errors import (ConfigurationError, EmptyDocumentError, InvalidFileError, LLMError,
                          classify_llm_error)
from utils.helpers import clean_text, format_size, make_doc_id, truncate
from utils.pdf_utils import PageText, extract_pages
from tests.conftest import FakeLLM, make_pdf

RESUME = ["Skills: Python, SQL, Streamlit and machine learning projects.", "Education: B.Tech in Computer Science."]
JOB = ["Requirements: Kubernetes, Docker, Terraform and cloud infrastructure experience."]


def test_clean_text_rejoins_hyphenation_and_lines():
    assert clean_text("inter-\nnational line one\nline two\n\n\n\nNext") == "international line one line two\n\nNext"


def test_helpers():
    assert format_size(2048) == "2.0 KB"
    assert make_doc_id(b"a") == make_doc_id(b"a") != make_doc_id(b"b")
    assert len(truncate("word " * 100, 20)) <= 21


def test_extract_pages_valid():
    result = extract_pages(make_pdf(RESUME))
    assert result.page_count == 2 and [p.page_number for p in result.pages] == [1, 2]


def test_extract_pages_errors():
    with pytest.raises(InvalidFileError):
        extract_pages(b"")
    with pytest.raises(InvalidFileError):
        extract_pages(b"not a pdf at all")
    with pytest.raises(InvalidFileError):
        extract_pages(b"%PDF-1.4 garbage garbage")
    with pytest.raises(EmptyDocumentError):
        extract_pages(make_pdf(["", ""]))


def test_blank_pages_are_skipped_but_counted():
    result = extract_pages(make_pdf(["Hello world content", ""]))
    assert result.page_count == 2 and len(result.pages) == 1


def test_chunk_metadata_and_size():
    chunks = split_pages([PageText(3, "Sentence number one. " * 60)], "doc1", "a.pdf", 300, 50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {3}
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_ingest_and_retrieve(store, settings):
    info = ingest_pdf(store, "resume.pdf", make_pdf(RESUME), settings)
    assert info.page_count == 2 and info.chunk_count >= 2
    hits = retrieve(store, "What Python skills are listed?", [info.doc_id], 4, 0.2)
    assert hits and hits[0].page == 1 and hits[0].filename == "resume.pdf"


def test_retrieval_is_scoped_to_selected_documents(store, settings):
    a = ingest_pdf(store, "resume.pdf", make_pdf(RESUME), settings)
    b = ingest_pdf(store, "job.pdf", make_pdf(JOB), settings)
    hits = retrieve(store, "Kubernetes Docker Terraform", [a.doc_id], 4, 0.0)
    assert {h.filename for h in hits} == {"resume.pdf"}
    both = retrieve(store, "skills and requirements", [a.doc_id, b.doc_id], 4, 0.0)
    assert {h.filename for h in both} == {"resume.pdf", "job.pdf"}


def test_delete_document(store, settings):
    a = ingest_pdf(store, "resume.pdf", make_pdf(RESUME), settings)
    store.delete_document(a.doc_id)
    assert retrieve(store, "Python", [a.doc_id], 4, 0.0) == []


def test_answer_uses_context_and_returns_sources(store, settings):
    a = ingest_pdf(store, "resume.pdf", make_pdf(RESUME), settings)
    llm = FakeLLM("Python and SQL.")
    answer = answer_question("Which Python skills are listed?", store, [a.doc_id], settings, llm)
    assert answer.found and answer.sources and answer.text == "Python and SQL."
    prompt_text = " ".join(m.content for m in llm.calls[0])
    assert "<context>" in prompt_text and "Streamlit" in prompt_text and "ONLY" in prompt_text


def test_no_relevant_chunks_skips_llm(store, settings):
    a = ingest_pdf(store, "resume.pdf", make_pdf(RESUME), settings)
    llm = FakeLLM()
    answer = answer_question("zzzz qqqq xxxx", store, [a.doc_id], settings, llm)
    assert not answer.found and answer.text == NOT_FOUND_MESSAGE and llm.calls == []


def test_model_not_found_reply_hides_sources(store, settings):
    a = ingest_pdf(store, "resume.pdf", make_pdf(RESUME), settings)
    answer = answer_question("Python skills?", store, [a.doc_id], settings, FakeLLM(NOT_FOUND_MESSAGE))
    assert not answer.found and answer.sources == []


def test_missing_api_key(store, settings):
    from dataclasses import replace
    a = ingest_pdf(store, "resume.pdf", make_pdf(RESUME), settings)
    with pytest.raises(ConfigurationError):
        answer_question("Python skills?", store, [a.doc_id], replace(settings, google_api_key=""))


def test_summary_modes(store, settings):
    a = ingest_pdf(store, "resume.pdf", make_pdf(RESUME), settings)
    llm = FakeLLM("- point")
    for mode in ("short", "key_points", "findings"):
        assert summarize(mode, store, [a.doc_id], settings, llm) == "- point"


def test_empty_llm_reply_is_friendly_error(store, settings):
    a = ingest_pdf(store, "resume.pdf", make_pdf(RESUME), settings)
    with pytest.raises(LLMError):
        summarize("short", store, [a.doc_id], settings, FakeLLM(""))


@pytest.mark.parametrize("message,expected", [
    ("400 API key not valid. Please pass a valid API key.", ConfigurationError),
    ("429 RESOURCE_EXHAUSTED quota exceeded", LLMError),
    ("httpx.ConnectError: connection failed", LLMError),
])
def test_classify_llm_error(message, expected):
    err = classify_llm_error(Exception(message))
    assert isinstance(err, expected) and "Traceback" not in str(err)
