"""DocuMind AI: Streamlit entry point."""
from __future__ import annotations

import logging

import streamlit as st

from components import state
from components.chat import render_chat, render_summary
from components.ui import (inject_css, render_brand, render_document_list, render_empty_state, render_features,
                           render_hero, render_how_it_works, render_stats)
from components.upload import upload_button
from config.settings import get_settings
from utils.errors import DocuMindError
from utils.helpers import configure_logging

st.set_page_config(page_title="DocuMind AI", page_icon=":material/description:", layout="wide")
configure_logging()
logger = logging.getLogger(__name__)
inject_css()
state.init_state()
settings = get_settings()


def remove_document(doc_id: str) -> None:
    try:
        state.get_store(settings).delete_document(doc_id)
    except DocuMindError as exc:
        state.add_notice("error", str(exc))
        return
    state.unregister_document(doc_id)
    state.add_notice("success", "Document removed.")
    st.rerun()


def render_sidebar(docs: dict) -> list[str]:
    with st.sidebar:
        render_brand()
        st.write("")
        if docs:
            upload_button("sidebar_upload", "Add documents", primary=False)
            if st.session_state.pop("sync_scope", False) or "scope_widget" not in st.session_state:
                st.session_state.scope_widget = list(st.session_state.scope)
            st.multiselect(
                "Search in", options=list(docs), key="scope_widget", format_func=lambda d: docs[d].filename,
                help="Questions and summaries use only the selected documents.",
            )
            st.session_state.scope = list(st.session_state.scope_widget)
            if st.button("Clear conversation", icon=":material/delete_sweep:", use_container_width=True,
                         disabled=not st.session_state.messages):
                st.session_state.messages = []
                st.rerun()
        st.divider()
        if settings.has_api_key:
            st.caption(f"Model: {settings.gemini_model}  \nAPI key: configured")
        else:
            st.warning("No Gemini API key found. Add GOOGLE_API_KEY to your .env file or Streamlit secrets.")
    return st.session_state.scope


def render_landing() -> None:
    render_hero()
    upload_button("hero_upload")
    st.write("")
    render_features()
    render_how_it_works()
    render_empty_state()


def render_workspace(docs: dict, scope: list[str]) -> None:
    render_stats(docs, [docs[d].filename for d in scope])
    view = st.segmented_control("View", ["Chat", "Summary", "Documents"], key="view", label_visibility="collapsed")
    if view == "Documents":
        render_document_list(docs, remove_document)
    elif not scope:
        st.info("Select at least one document in the sidebar to continue.")
    elif view == "Summary":
        render_summary(scope)
    else:
        render_chat(scope)


docs = state.docs()
active_scope = render_sidebar(docs)
state.show_notices()
if docs:
    render_workspace(docs, active_scope)
else:
    render_landing()
