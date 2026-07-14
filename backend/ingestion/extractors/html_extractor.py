"""Extractor for HTML files/pages."""

from __future__ import annotations

from backend.core.exceptions import ValidationAppError
from backend.ingestion.extractors.base import ExtractedDocument, Extractor

# Tags whose content is markup/code, never document prose.
_NON_CONTENT_TAGS = ("script", "style", "noscript", "template", "svg")


class HTMLExtractor(Extractor):
    def extract(self, raw_bytes: bytes) -> ExtractedDocument:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ValidationAppError(
                "HTML ingestion requires the 'beautifulsoup4' package"
            ) from exc

        html = raw_bytes.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(_NON_CONTENT_TAGS):
            tag.decompose()

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        author = None
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta and author_meta.get("content"):
            author = author_meta["content"].strip()

        # Insert a newline between block-level elements so words from adjacent
        # tags don't get concatenated (get_text() alone would run them together).
        text = soup.get_text(separator="\n")

        return ExtractedDocument(text=text, title=title, author=author)
