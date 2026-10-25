"""Upload dialog: validate files, show their status, and run the ingestion pipeline."""
from __future__ import annotations

import logging

import streamlit as st

from components import state
from config.settings import Settings, get_settings
from rag.ingest import ingest_pdf
from utils.errors import DocuMindError
from utils.helpers import format_size, make_doc_id

logger = logging.getLogger(__name__)


def _check(file, settings: Settings, seen: set[str]) -> tuple[str, bool, str]:
    """Return (status text, is_valid, doc_id) for one uploaded file."""
    data = file.getvalue()
    if not file.name.lower().endswith(".pdf") or not data.lstrip()[:5] == b"%PDF-":
        return "Not a valid PDF", False, ""
    if len(data) > settings.max_file_mb * 1024 * 1024:
        return f"Larger than {settings.max_file_mb} MB", False, ""
    doc_id = make_doc_id(data)
    if doc_id in state.docs() or doc_id in seen:
        return "Already added", False, doc_id
    return "Ready to process", True, doc_id


def _process(files: list, settings: Settings) -> None:
    progress = st.progress(0.0, text="Preparing…")
    store = None
    with st.spinner("Loading the embedding model (the first run downloads it)…"):
        try:
            store = state.get_store(settings)
        except DocuMindError as exc:
            state.add_notice("error", str(exc))
    if store is not None:
        for i, file in enumerate(files, start=1):
            progress.progress((i - 1) / len(files), text=f"Processing {file.name} ({i}/{len(files)})")
            try:
                info = ingest_pdf(store, file.name, file.getvalue(), settings)
                state.register_document(info)
                state.add_notice("success", f"{file.name}: {info.page_count} pages, {info.chunk_count} chunks indexed.")
            except DocuMindError as exc:
                state.add_notice("error", f"{file.name}: {exc}")
            except Exception:
                logger.exception("Unexpected failure while processing %s", file.name)
                state.add_notice("error", f"{file.name}: something went wrong while processing this file.")
    st.session_state.uploader_nonce += 1
    st.rerun()


@st.dialog("Upload documents", width="large")
def upload_dialog() -> None:
    settings = get_settings()
    remaining = settings.max_documents - len(state.docs())
    if remaining <= 0:
        st.info(f"You have reached the limit of {settings.max_documents} documents. Remove one to add another.")
        return
    st.caption(f"PDF files up to {settings.max_file_mb} MB. You can add {remaining} more document(s).")
    files = st.file_uploader(
        "PDF files", type=["pdf"], accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_nonce}", label_visibility="collapsed",
    )
    valid, seen = [], set()
    for file in files or []:
        status, ok, doc_id = _check(file, settings, seen)
        if ok and len(valid) >= remaining:
            status, ok = "Over the document limit", False
        if ok:
            valid.append(file)
            seen.add(doc_id)
        with st.container(border=True):
            left, right = st.columns([3, 2])
            left.markdown(f"**{file.name}**  \n<span class='dm-muted'>{format_size(file.size)}</span>", unsafe_allow_html=True)
            right.markdown(f"<span class='dm-badge {'dm-ok' if ok else 'dm-warn'}'>{status}</span>", unsafe_allow_html=True)
    if not files:
        st.markdown("<div class='dm-muted'>Select one or more PDFs to begin.</div>", unsafe_allow_html=True)
    if st.button("Process documents", type="primary", icon=":material/play_arrow:", disabled=not valid, use_container_width=True):
        _process(valid, settings)


def upload_button(key: str, label: str = "Upload documents", primary: bool = True) -> None:
    if st.button(label, key=key, type="primary" if primary else "secondary", icon=":material/upload_file:"):
        upload_dialog()
