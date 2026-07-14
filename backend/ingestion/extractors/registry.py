"""Maps MIME content types (and file-extension fallbacks) to format extractors."""

from __future__ import annotations

from pathlib import PurePosixPath

from backend.core.exceptions import ValidationAppError
from backend.ingestion.extractors.base import Extractor
from backend.ingestion.extractors.docx_extractor import DocxExtractor
from backend.ingestion.extractors.excel_extractor import ExcelExtractor
from backend.ingestion.extractors.html_extractor import HTMLExtractor
from backend.ingestion.extractors.pdf_extractor import PDFExtractor
from backend.ingestion.extractors.txt_extractor import TextExtractor

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

SUPPORTED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/html",
        "application/pdf",
        DOCX_MIME,
        XLSX_MIME,
    }
)

# Clients (browsers, curl, some SDKs) frequently send a generic or wrong MIME type
# for office documents — fall back to the file extension when that happens.
_EXTENSION_TO_CONTENT_TYPE: dict[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".htm": "text/html",
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".docx": DOCX_MIME,
    ".xlsx": XLSX_MIME,
}

_EXTRACTORS: dict[str, Extractor] = {
    "text/plain": TextExtractor(),
    "text/markdown": TextExtractor(),
    "text/html": HTMLExtractor(),
    "application/pdf": PDFExtractor(),
    DOCX_MIME: DocxExtractor(),
    XLSX_MIME: ExcelExtractor(),
}


def resolve_content_type(content_type: str | None, filename: str) -> str:
    """Return a supported content type for this upload, or raise if none applies.

    Trusts the client-supplied `content_type` when it is one we support; otherwise
    falls back to the file extension (handles generic `application/octet-stream`
    uploads for office documents).
    """
    if content_type in SUPPORTED_CONTENT_TYPES:
        return content_type

    extension = PurePosixPath(filename or "").suffix.lower()
    resolved = _EXTENSION_TO_CONTENT_TYPE.get(extension)
    if resolved is not None:
        return resolved

    raise ValidationAppError(
        f"Unsupported file '{filename}' (content type '{content_type}'). "
        f"Supported formats: PDF, DOCX, XLSX, TXT, Markdown, HTML."
    )


def get_extractor(content_type: str) -> Extractor:
    extractor = _EXTRACTORS.get(content_type)
    if extractor is None:
        raise ValidationAppError(f"Unsupported content type: {content_type}")
    return extractor
