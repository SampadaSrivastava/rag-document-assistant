"""PDF validation and text extraction (PyMuPDF)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pymupdf

from utils.errors import EmptyDocumentError, InvalidFileError
from utils.helpers import clean_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based, matches what a reader sees
    text: str


@dataclass(frozen=True)
class ExtractedPDF:
    pages: list[PageText]  # only pages that contain text
    page_count: int        # all pages, including blank/scanned ones


def extract_pages(data: bytes) -> ExtractedPDF:
    """Extract cleaned text per page. Raises a friendly DocuMindError on any problem."""
    if not data:
        raise InvalidFileError("The file is empty.")
    if not data.lstrip()[:5] == b"%PDF-":
        raise InvalidFileError("This file is not a valid PDF.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        logger.warning("PyMuPDF could not open file: %s", exc)
        raise InvalidFileError("This PDF is corrupted and could not be opened.") from exc

    with doc:
        if doc.needs_pass:
            raise InvalidFileError("This PDF is password-protected. Remove the password and upload it again.")
        pages: list[PageText] = []
        try:
            for number, page in enumerate(doc, start=1):
                text = clean_text(page.get_text("text"))
                if text:
                    pages.append(PageText(number, text))
        except Exception as exc:
            logger.warning("Failed while reading pages: %s", exc)
            raise InvalidFileError("This PDF appears to be damaged and could not be read.") from exc
        page_count = doc.page_count

    if not pages:
        raise EmptyDocumentError(
            "No selectable text was found. This looks like a scanned PDF; run OCR on it first and upload the result."
        )
    return ExtractedPDF(pages=pages, page_count=page_count)
