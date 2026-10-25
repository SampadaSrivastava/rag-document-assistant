"""Chat and summary views."""
from __future__ import annotations

import logging

import streamlit as st

from components import state
from components.sources import render_sources
from config.settings import Settings, get_settings
from rag.qa_chain import answer_question, summarize
from utils.errors import DocuMindError

logger = logging.getLogger(__name__)

SUGGESTIONS = ["Summarize this document.", "What are the key points?", "What are the main conclusions?"]
MULTI_DOC_SUGGESTION = "How do these documents differ?"
SUMMARY_MODES = {"short": "Short summary", "key_points": "Key points", "findings": "Important findings"}
USER_ICON, ASSISTANT_ICON = ":material/person:", ":material/description:"


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"], avatar=USER_ICON if message["role"] == "user" else ASSISTANT_ICON):
        if message.get("error"):
            st.error(message["content"])
            return
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


def _ask(question: str, scope: list[str], settings: Settings) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    _render_message(st.session_state.messages[-1])
    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        with st.spinner("Searching your documents…"):
            try:
                answer = answer_question(question, state.get_store(settings), scope, settings)
                reply = {"role": "assistant", "content": answer.text, "sources": answer.sources}
            except DocuMindError as exc:
                reply = {"role": "assistant", "content": str(exc), "error": True}
            except Exception:
                logger.exception("Unexpected failure while answering")
                reply = {"role": "assistant", "content": "Something went wrong while answering. Please try again.", "error": True}
    st.session_state.messages.append(reply)
    st.rerun()


def render_chat(scope: list[str]) -> None:
    settings = get_settings()
    typed = st.chat_input("Ask a question about your documents", disabled=not scope)
    question = typed or st.session_state.pop("queued_question", None)

    if not st.session_state.messages and not question:
        st.markdown("<div class='dm-label'>Suggested questions</div>", unsafe_allow_html=True)
        options = SUGGESTIONS + ([MULTI_DOC_SUGGESTION] if len(scope) > 1 else [])
        columns = st.columns(len(options))
        for i, (col, text) in enumerate(zip(columns, options)):
            if col.button(text, key=f"suggest_{i}", use_container_width=True):
                st.session_state.queued_question = text
                st.rerun()
    for message in st.session_state.messages:
        _render_message(message)
    if question:
        _ask(question, scope, settings)


def render_summary(scope: list[str]) -> None:
    settings = get_settings()
    st.markdown("Generate an overview of the selected documents. Summaries are built from excerpts sampled evenly across the text.")
    columns = st.columns(len(SUMMARY_MODES))
    for col, (mode, label) in zip(columns, SUMMARY_MODES.items()):
        if col.button(label, key=f"sum_{mode}", use_container_width=True, disabled=not scope):
            key = (mode, tuple(scope))
            if key not in st.session_state.summaries:
                with st.spinner(f"Generating {label.lower()}…"):
                    try:
                        st.session_state.summaries[key] = summarize(mode, state.get_store(settings), scope, settings)
                    except DocuMindError as exc:
                        st.error(str(exc))
                    except Exception:
                        logger.exception("Unexpected failure while summarising")
                        st.error("Something went wrong while generating the summary. Please try again.")
            st.session_state.active_summary = key
    key = st.session_state.get("active_summary")
    if key in st.session_state.summaries:
        with st.container(border=True):
            st.markdown(f"<div class='dm-label'>{SUMMARY_MODES[key[0]]}</div>", unsafe_allow_html=True)
            st.markdown(st.session_state.summaries[key])
