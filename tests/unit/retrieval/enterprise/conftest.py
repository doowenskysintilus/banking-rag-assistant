"""Shared fixtures for enterprise retrieval tests: an in-memory SQLite DB seeded
with a user + documents/chunks, plus fakes for the vector store and embedding
service so these tests never touch a real model or index.
"""

from __future__ import annotations

import uuid
import zlib
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.core.database import Base
from backend.models.document import Document, DocumentChunk
from backend.models.user import User
from backend.retrieval.vector_store.filters import matches
from backend.retrieval.vector_store.types import SearchResult, VectorStoreBase


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = __import__(
        "sqlalchemy.ext.asyncio", fromlist=["async_sessionmaker"]
    ).async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        yield session

    await engine.dispose()


async def make_user(db: AsyncSession, *, user_id: str | None = None) -> User:
    user = User(
        id=user_id or str(uuid.uuid4()),
        email=f"{uuid.uuid4().hex}@example.com",
        full_name="Test User",
        hashed_password="hashed",
    )
    db.add(user)
    await db.flush()
    return user


async def make_document(
    db: AsyncSession,
    *,
    owner_id: str,
    filename: str = "policy.pdf",
    is_current: bool = True,
    created_at: datetime | None = None,
) -> Document:
    document = Document(
        owner_id=owner_id,
        filename=filename,
        content_type="application/pdf",
        size_bytes=100,
        extracted_text="irrelevant",
        document_key=filename.lower(),
        content_hash=uuid.uuid4().hex,
        content_simhash=0,
        language="en",
        is_current=is_current,
    )
    db.add(document)
    await db.flush()
    if created_at is not None:
        document.created_at = created_at
    await db.flush()
    return document


async def make_chunk(
    db: AsyncSession,
    *,
    document: Document,
    content: str,
    vector_id: int,
    header_path: str = "",
) -> DocumentChunk:
    chunk = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=document.id,
        chunk_index=0,
        content=content,
        header_path=header_path,
        vector_id=vector_id,
    )
    db.add(chunk)
    await db.flush()
    return chunk


def iso_days_ago(days: float) -> str:
    from datetime import timedelta

    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class FakeVectorStore(VectorStoreBase):
    """In-memory VectorStoreBase — enough surface for the enterprise retrieval
    tests without pulling in real FAISS.
    """

    def __init__(self) -> None:
        self._vectors: dict[int, np.ndarray] = {}
        self._metadata: dict[int, dict] = {}

    async def add(self, vector_ids, vectors, metadatas=None, texts=None) -> None:
        for i, vector_id in enumerate(vector_ids):
            self._vectors[vector_id] = vectors[i]
            self._metadata[vector_id] = dict(metadatas[i]) if metadatas else {}

    async def update(
        self, vector_ids, vectors=None, metadatas=None, texts=None
    ) -> None:
        for i, vector_id in enumerate(vector_ids):
            if vectors is not None:
                self._vectors[vector_id] = vectors[i]
            if metadatas is not None:
                self._metadata[vector_id] = dict(metadatas[i])

    async def delete(self, vector_ids) -> None:
        for vector_id in vector_ids:
            self._vectors.pop(vector_id, None)
            self._metadata.pop(vector_id, None)

    async def search(self, query_vector, top_k, filters=None) -> list[SearchResult]:
        candidates = []
        for vector_id, vector in self._vectors.items():
            metadata = self._metadata.get(vector_id, {})
            if filters and not matches(metadata, filters):
                continue
            denom = (np.linalg.norm(query_vector) * np.linalg.norm(vector)) or 1.0
            score = float(np.dot(query_vector, vector) / denom)
            candidates.append(
                SearchResult(vector_id=vector_id, score=score, metadata=metadata)
            )
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[:top_k]

    async def hybrid_search(
        self, query_vector, query_text, top_k, filters=None
    ) -> list[SearchResult]:
        return await self.search(query_vector, top_k, filters)

    async def save(self) -> None:
        return None


class FakeEmbeddingService:
    """Deterministic embedder: encodes text length so tests can construct
    predictable similarity relationships without a real model.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        return (
            np.stack([self._encode(t) for t in texts])
            if texts
            else np.empty((0, self.dim), dtype="float32")
        )

    async def embed_query(self, text: str) -> np.ndarray:
        return self._encode(text)

    def _encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        # zlib.crc32 (not the builtin hash()) so results are stable across runs/
        # processes — Python randomizes str hash() per-process by default.
        vec[zlib.crc32(text.encode("utf-8")) % self.dim] = 1.0
        return vec
