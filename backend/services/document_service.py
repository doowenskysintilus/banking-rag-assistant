"""Document upload, listing, deletion, and version history.

Upload flow:
    1. Read + size-check the raw bytes.
    2. Run the ingestion pipeline synchronously (extract -> clean -> detect
       language -> hash -> build metadata). CPU-bound, so it's offloaded to a
       thread to avoid blocking the event loop.
    3. Decide NEW / NEW_VERSION / EXACT_DUPLICATE by comparing against this
       owner's existing *current* documents (backend.ingestion.versioning).
    4. Persist the raw file (audit trail) and the Document row. If this is a
       new version, retire the superseded version's vectors/chunks.
    5. Schedule the background task that chunks + embeds + indexes the
       already-extracted text.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import Settings
from backend.core.exceptions import ConflictError, NotFoundError, ValidationAppError
from backend.core.logging import get_logger
from backend.ingestion.dedup import find_near_duplicate
from backend.ingestion.pipeline import get_ingestion_pipeline
from backend.ingestion.versioning import DocumentRef, IngestionDecision, decide
from backend.models.document import Document, DocumentChunk, DocumentStatus
from backend.retrieval.vector_store import get_vector_store
from backend.workers.tasks import ingest_document

logger = get_logger(__name__)

_NEAR_DUPLICATE_THRESHOLD = 3


class DocumentService:
    def __init__(self, db: AsyncSession, settings: Settings) -> None:
        self._db = db
        self._settings = settings
        self._storage_dir = Path(settings.document_storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    async def upload(
        self, owner_id: str, file: UploadFile, background_tasks: BackgroundTasks
    ) -> Document:
        raw_bytes = await file.read()
        max_bytes = self._settings.max_upload_size_mb * 1024 * 1024
        if len(raw_bytes) > max_bytes:
            raise ValidationAppError(
                f"File exceeds the {self._settings.max_upload_size_mb}MB upload limit"
            )

        filename = file.filename or "untitled"
        pipeline = get_ingestion_pipeline()
        processed = await asyncio.to_thread(
            pipeline.process, filename, file.content_type, raw_bytes
        )

        exact_duplicate = await self._find_exact_duplicate(
            owner_id, processed.metadata.content_hash
        )
        same_key_current = await self._find_current_by_key(
            owner_id, processed.document_key
        )
        versioning_result = decide(
            exact_duplicate_candidate=exact_duplicate,
            same_key_current_candidate=same_key_current,
        )

        if versioning_result.decision is IngestionDecision.EXACT_DUPLICATE:
            raise ConflictError(
                "An identical document already exists in the knowledge base",
                details={"existing_document_id": versioning_result.duplicate_of.id},
            )

        near_duplicate_id = await self._find_near_duplicate(
            owner_id, processed.document_key, processed.metadata.simhash
        )

        document = Document(
            owner_id=owner_id,
            filename=filename,
            content_type=processed.resolved_content_type,
            size_bytes=len(raw_bytes),
            status=DocumentStatus.PENDING,
            extracted_text=processed.cleaned_text,
            document_key=processed.document_key,
            content_hash=processed.metadata.content_hash,
            content_simhash=processed.metadata.simhash,
            language=processed.metadata.language,
            doc_metadata=processed.metadata.model_dump_json(),
            near_duplicate_of_id=near_duplicate_id,
        )

        if versioning_result.decision is IngestionDecision.NEW_VERSION:
            previous = await self._db.get(Document, versioning_result.supersedes.id)
            document.version = versioning_result.next_version
            document.previous_version_id = previous.id
            await self._retire_document(previous)

        self._db.add(document)
        await self._db.flush()

        storage_path = self._storage_dir / f"{document.id}_{uuid.uuid4().hex}"
        storage_path.write_bytes(raw_bytes)

        await self._db.commit()
        await self._db.refresh(document)

        background_tasks.add_task(ingest_document, document.id)
        logger.info(
            "document_upload_accepted",
            extra={
                "document_id": document.id,
                "version": document.version,
                "language": document.language,
                "decision": versioning_result.decision.value,
            },
        )
        return document

    async def list_documents(
        self, owner_id: str, limit: int = 50, offset: int = 0
    ) -> tuple[list[Document], int]:
        items_result = await self._db.execute(
            select(Document)
            .where(Document.owner_id == owner_id, Document.is_current.is_(True))
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        count_result = await self._db.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.owner_id == owner_id, Document.is_current.is_(True))
        )
        return list(items_result.scalars().all()), int(count_result.scalar_one())

    async def list_versions(self, owner_id: str, document_id: str) -> list[Document]:
        """Full version history of a document's logical lineage, newest first."""
        head = await self.get_document(owner_id, document_id)

        result = await self._db.execute(
            select(Document)
            .where(
                Document.owner_id == owner_id,
                Document.document_key == head.document_key,
            )
            .order_by(Document.version.desc())
        )
        return list(result.scalars().all())

    async def get_document(self, owner_id: str, document_id: str) -> Document:
        document = await self._db.get(Document, document_id)
        if document is None or document.owner_id != owner_id:
            raise NotFoundError(f"Document {document_id} was not found")
        return document

    async def delete_document(self, owner_id: str, document_id: str) -> None:
        document = await self.get_document(owner_id, document_id)
        await self._delete_vectors_and_chunks(document.id)
        await self._db.delete(document)
        await self._db.commit()
        self._invalidate_bm25_cache(owner_id)
        logger.info("document_deleted", extra={"document_id": document_id})

    async def _retire_document(self, document: Document) -> None:
        """Mark a superseded version as no longer current and drop its vectors.

        The row itself is kept (status/audit trail); only its embeddings and
        chunk rows are removed, since a superseded version must not be retrieved.
        """
        await self._delete_vectors_and_chunks(document.id)
        document.is_current = False
        self._invalidate_bm25_cache(document.owner_id)

    def _invalidate_bm25_cache(self, owner_id: str) -> None:
        if self._settings.rag_retrieval_engine != "enterprise":
            return
        from backend.retrieval.enterprise.bm25_corpus import invalidate_owner_bm25_cache

        invalidate_owner_bm25_cache(owner_id)

    async def _delete_vectors_and_chunks(self, document_id: str) -> None:
        chunk_result = await self._db.execute(
            select(DocumentChunk.vector_id).where(
                DocumentChunk.document_id == document_id
            )
        )
        vector_ids = [row[0] for row in chunk_result.all()]
        if vector_ids:
            vector_store = get_vector_store()
            await vector_store.delete(vector_ids)

        # Bulk-delete chunks explicitly rather than relying on the ORM cascade: an
        # implicit lazy-load of `.chunks` would raise MissingGreenlet under async
        # SQLAlchemy.
        await self._db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )

    async def _find_exact_duplicate(
        self, owner_id: str, content_hash: str
    ) -> DocumentRef | None:
        result = await self._db.execute(
            select(Document.id, Document.version).where(
                Document.owner_id == owner_id,
                Document.is_current.is_(True),
                Document.content_hash == content_hash,
            )
        )
        row = result.first()
        return DocumentRef(id=row[0], version=row[1]) if row else None

    async def _find_current_by_key(
        self, owner_id: str, document_key: str
    ) -> DocumentRef | None:
        result = await self._db.execute(
            select(Document.id, Document.version).where(
                Document.owner_id == owner_id,
                Document.is_current.is_(True),
                Document.document_key == document_key,
            )
        )
        row = result.first()
        return DocumentRef(id=row[0], version=row[1]) if row else None

    async def _find_near_duplicate(
        self, owner_id: str, document_key: str, simhash: int
    ) -> str | None:
        result = await self._db.execute(
            select(Document.id, Document.content_simhash).where(
                Document.owner_id == owner_id,
                Document.is_current.is_(True),
                Document.document_key != document_key,
            )
        )
        candidates = [(row[0], row[1]) for row in result.all()]
        return find_near_duplicate(
            simhash, candidates, threshold=_NEAR_DUPLICATE_THRESHOLD
        )
