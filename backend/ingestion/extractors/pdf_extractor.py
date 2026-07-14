"""Extractor for PDF files, using pypdf."""

from __future__ import annotations

import io

from backend.core.exceptions import ValidationAppError
from backend.ingestion.extractors.base import ExtractedDocument, Extractor


class PDFExtractor(Extractor):
    def extract(self, raw_bytes: bytes) -> ExtractedDocument:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
        except PdfReadError as exc:
            raise ValidationAppError(f"Could not parse PDF: {exc}") from exc

        pages_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages_text)

        info = reader.metadata
        title = getattr(info, "title", None) if info else None
        author = getattr(info, "author", None) if info else None

        return ExtractedDocument(
            text=text,
            title=title,
            author=author,
            page_count=len(reader.pages),
        )
