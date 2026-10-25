"""Question answering and summarisation: retrieve -> build prompt -> call Gemini."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from config.settings import Settings
from rag.prompts import NOT_FOUND_MESSAGE, QA_PROMPT, SUMMARY_PROMPT, SUMMARY_TASKS
from rag.retriever import RetrievedChunk, retrieve
from rag.vector_store import DocumentStore
from utils.errors import ConfigurationError, DocuMindError, LLMError, classify_llm_error

logger = logging.getLogger(__name__)

SUMMARY_CHUNK_LIMIT = 12


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    found: bool = True


@lru_cache(maxsize=4)
def get_llm(api_key: str, model: str) -> Any:
    """Create the Gemini client once per (key, model) instead of on every question."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2, max_retries=2, timeout=60)


def _message_text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [p if isinstance(p, str) else p.get("text", "") for p in content if isinstance(p, (str, dict))]
    return "".join(parts).strip()


def _generate(messages: list, settings: Settings, llm: Any | None) -> str:
    if llm is None:
        if not settings.has_api_key:
            raise ConfigurationError("No Gemini API key found. Add GOOGLE_API_KEY to your .env file or Streamlit secrets.")
        llm = get_llm(settings.google_api_key, settings.gemini_model)
    try:
        text = _message_text(llm.invoke(messages))
    except DocuMindError:
        raise
    except Exception as exc:
        logger.exception("Gemini request failed")
        raise classify_llm_error(exc) from exc
    if not text:
        raise LLMError("The model returned an empty response (it may have been blocked). Try rephrasing your question.")
    return text


def _format_context(items: list[tuple[str, int, str]]) -> str:
    return "\n\n".join(f"[Excerpt {i} | {name} | page {page}]\n{text}" for i, (name, page, text) in enumerate(items, 1))


def answer_question(
    question: str, store: DocumentStore, doc_ids: list[str], settings: Settings, llm: Any | None = None
) -> Answer:
    """Answer from retrieved chunks only. If nothing relevant is retrieved the LLM is not called at all."""
    chunks = retrieve(store, question, doc_ids, settings.top_k, settings.score_threshold)
    if not chunks:
        return Answer(NOT_FOUND_MESSAGE, [], found=False)
    context = _format_context([(c.filename, c.page, c.text) for c in chunks])
    text = _generate(QA_PROMPT.format_messages(context=context, question=question), settings, llm)
    if text.startswith(NOT_FOUND_MESSAGE):
        return Answer(NOT_FOUND_MESSAGE, [], found=False)
    return Answer(text, chunks, found=True)


def summarize(mode: str, store: DocumentStore, doc_ids: list[str], settings: Settings, llm: Any | None = None) -> str:
    """Summarise using chunks sampled evenly across the selected documents."""
    chunks: list[Document] = store.sample_chunks(doc_ids, SUMMARY_CHUNK_LIMIT)
    if not chunks:
        raise LLMError("There is no document content to summarise.")
    context = _format_context([(c.metadata["filename"], int(c.metadata["page"]), c.page_content) for c in chunks])
    return _generate(SUMMARY_PROMPT.format_messages(context=context, task=SUMMARY_TASKS[mode]), settings, llm)
