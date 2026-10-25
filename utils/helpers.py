"""Small, dependency-free helpers."""
from __future__ import annotations

import hashlib
import logging
import re

_API_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def make_doc_id(data: bytes) -> str:
    """Stable ID from file contents, so the same PDF uploaded twice is recognised as a duplicate."""
    return hashlib.sha256(data).hexdigest()[:16]


def clean_text(text: str) -> str:
    """Normalise raw PDF text while keeping paragraph breaks for the splitter."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)           # single newline -> space
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


class _RedactSecrets(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _API_KEY_PATTERN.sub("[REDACTED]", str(record.msg))
        if record.args:
            record.args = tuple(_API_KEY_PATTERN.sub("[REDACTED]", str(a)) for a in record.args)
        return True


def configure_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_documind_configured", False):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_RedactSecrets())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root._documind_configured = True  # type: ignore[attr-defined]
