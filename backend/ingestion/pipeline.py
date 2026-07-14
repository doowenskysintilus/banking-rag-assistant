"""Ingestion pipeline: extract → clean → detect language → hash → build metadata.

Pure and synchronous (no DB, no network) so it's cheap to unit test. Callers that
run in an async context (e.g. `DocumentService.upload`) should invoke `process()`
via `asyncio.to_thread` since parsing large PDFs/spreadsheets is CPU-bound.

This module deliberately stops short of deduplication/versioning decisions — those
require querying existing documents and belong in the service layer
(`backend.services.document_service`), which composes this pipeline's output with
`backend.ingestion.versioning.decide(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.core.exceptions import ValidationAppError
from backend.ingestion.cleaning import clean_text
from backend.ingestion.dedup import compute_content_hash, compute_simhash
from backend.ingestion.extractors.registry import get_extractor, resolve_content_type
from backend.ingestion.language import detect_language
from backend.ingestion.metadata import DocumentMetadata
from backend.ingestion.versioning import normalize_document_key

_MIN_TEXT_LENGTH = 10


@dataclass
class ProcessedDocument:
    cleaned_text: str
    metadata: DocumentMetadata
    document_key: str
    resolved_content_type: str


class IngestionPipeline:
    def __init__(self, min_text_length: int = _MIN_TEXT_LENGTH) -> None:
        self._min_text_length = min_text_length

    def process(
        self, filename: str, content_type: str | None, raw_bytes: bytes
    ) -> ProcessedDocument:
        resolved_type = resolve_content_type(content_type, filename)
        extractor = get_extractor(resolved_type)

        extracted = extractor.extract(raw_bytes)
        cleaned = clean_text(extracted.text)

        if len(cleaned) < self._min_text_length:
            raise ValidationAppError(
                "Document does not contain enough extractable text to be useful "
                "(it may be a scanned image, empty, or corrupted)."
            )

        language_result = detect_language(cleaned)
        content_hash = compute_content_hash(cleaned)
        simhash = compute_simhash(cleaned)

        metadata = DocumentMetadata(
            title=extracted.title,
            author=extracted.author,
            page_count=extracted.page_count,
            sheet_names=extracted.sheet_names,
            word_count=len(cleaned.split()),
            char_count=len(cleaned),
            language=language_result.code,
            language_confidence=language_result.confidence,
            content_hash=content_hash,
            simhash=simhash,
            extra=extracted.extra,
        )

        return ProcessedDocument(
            cleaned_text=cleaned,
            metadata=metadata,
            document_key=normalize_document_key(filename),
            resolved_content_type=resolved_type,
        )


_pipeline_singleton: IngestionPipeline | None = None


def get_ingestion_pipeline() -> IngestionPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = IngestionPipeline()
    return _pipeline_singleton
