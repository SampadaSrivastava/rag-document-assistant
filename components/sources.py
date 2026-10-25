"""Source citations shown under each assistant answer."""
from __future__ import annotations

from collections import OrderedDict

import streamlit as st

from rag.retriever import RetrievedChunk
from utils.helpers import truncate


def render_sources(sources: list[RetrievedChunk]) -> None:
    if not sources:
        return
    pages: OrderedDict[tuple[str, int], None] = OrderedDict()
    for chunk in sorted(sources, key=lambda c: (c.filename, c.page)):
        pages[(chunk.filename, chunk.page)] = None
    items = "".join(f"<li>{name} — Page {page}</li>" for name, page in pages)
    st.markdown(f"<div class='dm-sources'><div class='dm-label'>Sources</div><ul>{items}</ul></div>", unsafe_allow_html=True)
    with st.expander(f"View {len(sources)} excerpt{'s' if len(sources) != 1 else ''}"):
        for i, chunk in enumerate(sources, start=1):
            st.markdown(
                f"<div class='dm-excerpt'><div class='dm-muted'>{i}. {chunk.filename} — Page {chunk.page} · similarity {chunk.score:.2f}</div>"
                f"<blockquote>{truncate(chunk.text, 400)}</blockquote></div>",
                unsafe_allow_html=True,
            )
