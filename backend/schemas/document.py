import json
from datetime import datetime
from typing import Any

from pydantic import Field

from backend.models.document import Document, DocumentStatus
from backend.schemas.common import ORMModel


class DocumentMetadataPublic(ORMModel):
    title: str | None = None
    author: str | None = None
    page_count: int | None = None
    sheet_names: list[str] | None = None
    word_count: int | None = None
    char_count: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentPublic(ORMModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    error_message: str | None = None

    document_key: str
    content_hash: str
    version: int
    is_current: bool
    previous_version_id: str | None = None
    near_duplicate_of_id: str | None = None
    language: str | None = None
    metadata: DocumentMetadataPublic | None = None

    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_document(cls, document: Document) -> "DocumentPublic":
        """Build the response model, unpacking the JSON-encoded `doc_metadata` blob
        that pydantic's `from_attributes` can't map automatically.
        """
        metadata = None
        if document.doc_metadata:
            raw = json.loads(document.doc_metadata)
            metadata = DocumentMetadataPublic(
                title=raw.get("title"),
                author=raw.get("author"),
                page_count=raw.get("page_count"),
                sheet_names=raw.get("sheet_names"),
                word_count=raw.get("word_count"),
                char_count=raw.get("char_count"),
                extra=raw.get("extra") or {},
            )

        return cls(
            id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            size_bytes=document.size_bytes,
            status=document.status,
            error_message=document.error_message,
            document_key=document.document_key,
            content_hash=document.content_hash,
            version=document.version,
            is_current=document.is_current,
            previous_version_id=document.previous_version_id,
            near_duplicate_of_id=document.near_duplicate_of_id,
            language=document.language,
            metadata=metadata,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class DocumentList(ORMModel):
    items: list[DocumentPublic] = Field(default_factory=list)
    total: int
