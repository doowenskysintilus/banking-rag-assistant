"""Bridges between our SQL/vector-store world and LangChain's `Document` type.

Both helpers enforce the same multi-tenant boundary as
`backend.rag.engine.RAGEngine._retrieve_candidates`: every chunk returned is re-verified
against the database's `Document.owner_id`, not just whatever `owner_id` happens to
be recorded in the vector store's (or BM25's) local metadata.
"""

from __future__ import annotations

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.document import Document as DocumentModel
from backend.models.document import DocumentChunk
from backend.retrieval.vector_store.types import SearchResult


def _chunk_to_document(chunk: DocumentChunk) -> Document:
    return Document(
        page_content=chunk.content,
        metadata={
            "vector_id": chunk.vector_id,
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "header_path": chunk.header_path,
        },
    )


async def fetch_owner_documents(db: AsyncSession, owner_id: str) -> list[Document]:
    """The full current corpus for one owner — used to build that owner's BM25
    index. Scoping by owner here (rather than filtering afterward) means another
    tenant's text never even enters the lexical index in the first place.
    """
    stmt = (
        select(DocumentChunk)
        .join(DocumentModel, DocumentModel.id == DocumentChunk.document_id)
        .where(DocumentModel.owner_id == owner_id, DocumentModel.is_current.is_(True))
    )
    result = await db.execute(stmt)
    return [_chunk_to_document(chunk) for chunk in result.scalars().all()]


async def hydrate_search_results(
    db: AsyncSession, hits: list[SearchResult], owner_id: str
) -> list[Document]:
    """Turns dense-search hits (vector_id + score + vector-store metadata) into
    `Document`s with real content, joining against the database — and re-checking
    ownership there, in case the vector store's metadata sidecar ever drifted.
    """
    if not hits:
        return []

    vector_ids = [hit.vector_id for hit in hits]
    stmt = (
        select(DocumentChunk)
        .join(DocumentModel, DocumentModel.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.vector_id.in_(vector_ids), DocumentModel.owner_id == owner_id
        )
    )
    result = await db.execute(stmt)
    chunks_by_vector_id = {chunk.vector_id: chunk for chunk in result.scalars().all()}

    documents: list[Document] = []
    for hit in hits:
        chunk = chunks_by_vector_id.get(hit.vector_id)
        if chunk is None:
            # Stale vector id, or an owner mismatch the join above just caught —
            # either way, skip rather than leak.
            continue
        documents.append(
            Document(
                page_content=chunk.content,
                metadata={
                    **hit.metadata,
                    "vector_id": hit.vector_id,
                    "score": hit.score,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.id,
                    "header_path": chunk.header_path,
                },
            )
        )
    return documents
