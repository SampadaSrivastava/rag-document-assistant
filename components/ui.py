"""Styling and static layout sections."""
from __future__ import annotations

import streamlit as st

from rag.ingest import DocInfo
from utils.helpers import format_size

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, button, input, textarea { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }
.block-container { max-width: 1040px; padding-top: 2.2rem; padding-bottom: 5rem; }
h1, h2, h3 { letter-spacing: -0.015em; font-weight: 600; }
.dm-brand { font-weight: 600; font-size: 1.05rem; letter-spacing: -0.01em; }
.dm-brand span { color: #2F5D8A; }
.dm-hero { padding: 2.2rem 0 1.2rem; }
.dm-hero h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); line-height: 1.15; margin: 0 0 .7rem; max-width: 18ch; }
.dm-hero p { font-size: 1.05rem; color: #4B5563; max-width: 56ch; line-height: 1.6; margin: 0 0 1.2rem; }
.dm-eyebrow, .dm-label { font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: #6B7280; margin-bottom: .5rem; }
.dm-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: .4rem 0 1.6rem; }
.dm-card { border: 1px solid #E3E7EC; border-radius: 8px; padding: 1rem 1.1rem; background: #fff; transition: border-color .15s ease; }
.dm-card:hover { border-color: #B7C3D0; }
.dm-card svg { width: 20px; height: 20px; stroke: #2F5D8A; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; margin-bottom: .55rem; }
.dm-card h4 { margin: 0 0 .3rem; font-size: .95rem; font-weight: 600; }
.dm-card p { margin: 0; font-size: .87rem; color: #4B5563; line-height: 1.5; }
.dm-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 1.6rem; }
.dm-step { border-top: 2px solid #D5DCE4; padding-top: .6rem; font-size: .87rem; color: #4B5563; line-height: 1.5; }
.dm-step b { display: block; color: #1B1F24; font-weight: 600; margin-bottom: .15rem; }
.dm-empty { border: 1px dashed #C9D1DA; border-radius: 8px; padding: 1.6rem; text-align: center; color: #4B5563; background: #FAFBFC; }
.dm-empty b { display: block; color: #1B1F24; margin-bottom: .2rem; }
.dm-muted { color: #6B7280; font-size: .82rem; }
.dm-badge { display: inline-block; padding: .15rem .55rem; border-radius: 999px; font-size: .75rem; font-weight: 500; }
.dm-ok { background: #E8F1EA; color: #226236; }
.dm-warn { background: #FBEDEA; color: #9A3324; }
.dm-sources { margin-top: .6rem; padding-top: .6rem; border-top: 1px solid #E3E7EC; }
.dm-sources ul { margin: 0; padding-left: 1.1rem; font-size: .88rem; color: #374151; }
.dm-excerpt { margin-bottom: .8rem; }
.dm-excerpt blockquote { margin: .25rem 0 0; padding: .1rem 0 .1rem .8rem; border-left: 2px solid #C9D1DA; font-size: .87rem; color: #374151; }
[data-testid="stMetric"] { border: 1px solid #E3E7EC; border-radius: 8px; padding: .7rem .9rem; background: #fff; }
[data-testid="stMetricLabel"] p { font-size: .78rem; color: #6B7280; }
[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 600; }
.stButton > button { border-radius: 6px; font-weight: 500; transition: border-color .15s ease, background-color .15s ease; }
[data-testid="stSidebar"] { border-right: 1px solid #E3E7EC; }
@media (max-width: 640px) { .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem; } .dm-hero { padding-top: 1rem; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""

_ICONS = {
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/>',
}

FEATURES = [
    ("chat", "Ask questions about documents", "Query one or several PDFs in plain language, in a single conversation."),
    ("search", "Semantic search", "Passages are matched by meaning, not just by shared keywords."),
    ("layers", "Context-aware answers", "Gemini answers only from the passages retrieved for your question."),
    ("file", "Source citations", "Every answer lists its document and page, with the excerpts used."),
]

STEPS = [
    ("1. Upload", "PDF text is extracted page by page and cleaned."),
    ("2. Index", "Pages are split into chunks and stored as embeddings in ChromaDB."),
    ("3. Retrieve", "Your question is embedded and the closest chunks are fetched."),
    ("4. Answer", "Gemini writes a reply from those chunks, and the sources are shown."),
]


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_brand() -> None:
    st.markdown("<div class='dm-brand'>DocuMind <span>AI</span></div>", unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        "<div class='dm-hero'><div class='dm-eyebrow'>Document Q&amp;A</div>"
        "<h1>Ask your documents, get answers you can verify.</h1>"
        "<p>Upload PDFs and ask questions in plain language. Answers are generated from the text of your files "
        "and always cite the page they came from.</p></div>",
        unsafe_allow_html=True,
    )


def render_features() -> None:
    cards = "".join(
        f"<div class='dm-card'><svg viewBox='0 0 24 24'>{_ICONS[i]}</svg><h4>{t}</h4><p>{d}</p></div>" for i, t, d in FEATURES
    )
    st.markdown(f"<div class='dm-grid'>{cards}</div>", unsafe_allow_html=True)


def render_how_it_works() -> None:
    steps = "".join(f"<div class='dm-step'><b>{t}</b>{d}</div>" for t, d in STEPS)
    st.markdown(f"<div class='dm-label'>How it works</div><div class='dm-steps'>{steps}</div>", unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        "<div class='dm-label'>Your documents</div>"
        "<div class='dm-empty'><b>No documents yet</b>Upload a PDF to start asking questions. Files stay in this session only.</div>",
        unsafe_allow_html=True,
    )


def render_stats(docs: dict[str, DocInfo], scope_names: list[str]) -> None:
    pages = sum(d.page_count for d in docs.values())
    chunks = sum(d.chunk_count for d in docs.values())
    cols = st.columns(4)
    cols[0].metric("Documents", len(docs))
    cols[1].metric("Pages", pages)
    cols[2].metric("Chunks", chunks)
    cols[3].metric("Status", "Ready")
    st.markdown(
        f"<div class='dm-muted' style='margin:.5rem 0 1rem'>Searching in: <b>{', '.join(scope_names) or 'no document selected'}</b></div>",
        unsafe_allow_html=True,
    )


def render_document_list(docs: dict[str, DocInfo], on_remove) -> None:
    for info in docs.values():
        with st.container(border=True):
            left, right = st.columns([4, 1], vertical_alignment="center")
            left.markdown(
                f"**{info.filename}**  \n<span class='dm-muted'>{format_size(info.size_bytes)} · {info.page_count} pages · "
                f"{info.chunk_count} chunks</span> &nbsp;<span class='dm-badge dm-ok'>Ready</span>",
                unsafe_allow_html=True,
            )
            if right.button("Remove", key=f"rm_{info.doc_id}", icon=":material/delete:", use_container_width=True):
                on_remove(info.doc_id)
