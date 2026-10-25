"""Streamlit session state helpers."""
from __future__ import annotations

import uuid

import streamlit as st

from config.settings import Settings
from rag.embeddings import get_embeddings
from rag.ingest import DocInfo
from rag.vector_store import DocumentStore


def init_state() -> None:
    defaults = {
        "session_id": uuid.uuid4().hex[:12],
        "store": None,
        "docs": {},          # doc_id -> DocInfo
        "scope": [],         # doc_ids currently searched
        "messages": [],      # chat history: {"role", "content", "sources", "found"}
        "summaries": {},     # (mode, doc_ids) -> text
        "notices": [],       # (level, message) shown once on the next render
        "sync_scope": False, # push `scope` into the multiselect before it is created
        "uploader_nonce": 0,
        "view": "Chat",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_store(settings: Settings) -> DocumentStore:
    if st.session_state.store is None:
        st.session_state.store = DocumentStore(
            st.session_state.session_id, get_embeddings(settings.embedding_model), settings.chroma_dir
        )
    return st.session_state.store


def docs() -> dict[str, DocInfo]:
    return st.session_state.docs


def add_notice(level: str, message: str) -> None:
    st.session_state.notices.append((level, message))


def show_notices() -> None:
    for level, message in st.session_state.notices:
        getattr(st, level)(message)
    st.session_state.notices = []


def register_document(info: DocInfo) -> None:
    st.session_state.docs[info.doc_id] = info
    st.session_state.scope = [*st.session_state.scope, info.doc_id]
    st.session_state.sync_scope = True
    st.session_state.summaries = {}


def unregister_document(doc_id: str) -> None:
    st.session_state.docs.pop(doc_id, None)
    st.session_state.scope = [d for d in st.session_state.scope if d != doc_id]
    st.session_state.sync_scope = True
    st.session_state.summaries = {}
    st.session_state.messages = []
