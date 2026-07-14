"""Extractor for plain text and Markdown files."""

from __future__ import annotations

from backend.ingestion.extractors.base import ExtractedDocument, Extractor


class TextExtractor(Extractor):
    def extract(self, raw_bytes: bytes) -> ExtractedDocument:
        text = raw_bytes.decode("utf-8", errors="replace")
        return ExtractedDocument(text=text)
