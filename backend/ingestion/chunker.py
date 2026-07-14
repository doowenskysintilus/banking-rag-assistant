"""Splits document text into overlapping chunks suitable for embedding + retrieval."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """Greedy fixed-size character chunking with overlap.

    Simple and dependency-free; swap for a token-aware or semantic chunker if
    retrieval quality on long documents needs improvement.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = _normalize(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(normalized)
    step = chunk_size - chunk_overlap

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += step

    return chunks
