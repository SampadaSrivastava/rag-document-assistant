"""Application settings, read from environment variables (.env locally, secrets on Streamlit Cloud)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    gemini_model: str
    embedding_model: str
    chroma_dir: Path | None  # None means in-memory only
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    max_file_mb: int
    max_documents: int

    @property
    def has_api_key(self) -> bool:
        return bool(self.google_api_key)


def get_settings() -> Settings:
    persist = os.getenv("CHROMA_PERSIST_DIR", "data/chroma").strip()
    chroma_dir = None
    if persist:
        path = Path(persist)
        chroma_dir = path if path.is_absolute() else BASE_DIR / path
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip(),
        chroma_dir=chroma_dir,
        chunk_size=_int("CHUNK_SIZE", 1000),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        top_k=_int("TOP_K", 5),
        score_threshold=_float("SCORE_THRESHOLD", 0.2),
        max_file_mb=_int("MAX_FILE_MB", 25),
        max_documents=_int("MAX_DOCUMENTS", 5),
    )
