"""Structured metadata extracted/derived for every ingested document."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Everything known about a document after extraction — persisted as JSON
    on `Document.doc_metadata` and returned via the API.
    """

    title: str | None = None
    author: str | None = None
    page_count: int | None = None
    sheet_names: list[str] | None = None
    word_count: int
    char_count: int
    language: str
    language_confidence: float
    content_hash: str
    simhash: int
    extra: dict[str, Any] = Field(default_factory=dict)
