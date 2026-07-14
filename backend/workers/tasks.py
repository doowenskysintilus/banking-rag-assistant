"""Background document ingestion.

Runs as a FastAPI `BackgroundTask` after the upload endpoint responds. Opens its own
DB session because the request-scoped session is closed by the time this executes.
For higher-throughput deployments, swap the call site in the documents service to
dispatch this same `ingest_document` function onto Celery/RQ instead — its signature
takes only a `document_id`, so it is queue-agnostic.

Text extraction/cleaning already happened synchronously at upload time (see
`backend.services.document_service.DocumentService.upload` and
`backend.ingestion.pipeline.IngestionPipeline`) so that duplicate/version decisions
could be made before writing the Document row. This task chunks (structure-aware
recursive/semantic/hybrid — see `backend.ingestion.chunking`), embeds, and indexes
`document.extracted_text`.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from backend.core.config import get_settings
from backend.core.database import AsyncSessionLocal
from backend.core.logging import get_logger
from backend.ingestion.chunker import chunk_text
from backend.ingestion.chunking.hybrid_chunker import ChunkerConfig, HybridChunker
from backend.ingestion.chunking.types import Chunk, ChunkStrategy
from backend.models.document import Document, DocumentChunk, DocumentStatus
from backend.monitoring.metrics import DOCUMENT_INGESTION_TOTAL
from backend.retrieval.embeddings import get_embedding_service
from backend.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


async def _next_vector_id(db) -> int:
    result = await db.execute(
        select(DocumentChunk.vector_id)
        .order_by(DocumentChunk.vector_id.desc())
        .limit(1)
    )
    last = result.scalar_one_or_none()
    return (last or 0) + 1


def _fixed_baseline_chunks(
    text: str, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    raw_chunks = chunk_text(text, chunk_size=max_chars, chunk_overlap=overlap_chars)
    return [
        Chunk(
            text=raw,
            raw_text=raw,
            index=i,
            char_start=0,
            char_end=len(raw),
            header_path="",
            strategy=ChunkStrategy.FIXED,
            approx_tokens=max(1, len(raw) // 4),
        )
        for i, raw in enumerate(raw_chunks)
    ]


async def ingest_document(document_id: str) -> None:
    async with AsyncSessionLocal() as db:
        document = await db.get(Document, document_id)
        if document is None:
            logger.error(
                "ingestion_document_missing", extra={"document_id": document_id}
            )
            return

        document.status = DocumentStatus.PROCESSING
        await db.commit()

        try:
            # Constructing these can fail (missing optional dependency, unreadable FAISS
            # index, bad credentials) — that must land in the same failure path as any
            # other ingestion error, not crash the background task silently.
            settings = get_settings()
            embeddings = get_embedding_service()
            vector_store = get_vector_store()

            strategy = ChunkStrategy(settings.chunking_strategy)

            if strategy is ChunkStrategy.FIXED:
                chunks = _fixed_baseline_chunks(
                    document.extracted_text,
                    settings.chunking_max_chars,
                    settings.rag_chunk_overlap,
                )
            else:
                chunker = HybridChunker(
                    ChunkerConfig(
                        strategy=strategy,
                        max_chars=settings.chunking_max_chars,
                        min_chars=settings.chunking_min_chars,
                        overlap_sentences=settings.chunking_overlap_sentences,
                        semantic_breakpoint_percentile=settings.chunking_semantic_breakpoint_percentile,
                        semantic_min_units=settings.chunking_semantic_min_units,
                    )
                )
                # RECURSIVE never needs embeddings; SEMANTIC/HYBRID use the same
                # embedding model that will later index the chunks.
                embedder = (
                    embeddings if strategy is not ChunkStrategy.RECURSIVE else None
                )
                chunks = await chunker.chunk(document.extracted_text, embedder=embedder)

            if not chunks:
                raise ValueError("No extractable text found in document")

            vectors = await embeddings.embed_texts([chunk.text for chunk in chunks])
            start_id = await _next_vector_id(db)
            vector_ids = list(range(start_id, start_id + len(chunks)))
            # Generated up front (rather than left to the DB's default=) so the same
            # id can be attached as vector metadata below, in the same pass.
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]

            metadatas = [
                {
                    "owner_id": document.owner_id,
                    "document_id": document.id,
                    "chunk_id": chunk_id,
                    "language": document.language,
                    "header_path": chunk.header_path,
                    # Powers the enterprise retrieval engine's recency boost — see
                    # backend.retrieval.enterprise.intelligent_ranker.
                    "created_at": document.created_at.isoformat(),
                }
                for chunk_id, chunk in zip(chunk_ids, chunks, strict=True)
            ]
            texts = [chunk.text for chunk in chunks]

            await vector_store.add(
                vector_ids, vectors, metadatas=metadatas, texts=texts
            )

            for chunk, vector_id, chunk_id in zip(
                chunks, vector_ids, chunk_ids, strict=True
            ):
                db.add(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document.id,
                        chunk_index=chunk.index,
                        content=chunk.text,
                        header_path=chunk.header_path,
                        vector_id=vector_id,
                    )
                )

            document.status = DocumentStatus.INDEXED
            document.error_message = None
            await db.commit()

            if settings.rag_retrieval_engine == "enterprise":
                # The owner's BM25 corpus is cached (see backend.retrieval.enterprise
                # .bm25_corpus) — invalidate it now instead of waiting out the TTL.
                from backend.retrieval.enterprise.bm25_corpus import (
                    invalidate_owner_bm25_cache,
                )

                invalidate_owner_bm25_cache(document.owner_id)

            DOCUMENT_INGESTION_TOTAL.labels("success").inc()
            logger.info(
                "document_ingested",
                extra={
                    "document_id": document_id,
                    "chunks": len(chunks),
                    "strategy": strategy.value,
                },
            )

        except Exception as exc:  # noqa: BLE001 - persist failure state, never crash the worker
            await db.rollback()
            document = await db.get(Document, document_id)
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.error_message = str(exc)[:2000]
                await db.commit()

            DOCUMENT_INGESTION_TOTAL.labels("failure").inc()
            logger.exception(
                "document_ingestion_failed", extra={"document_id": document_id}
            )
