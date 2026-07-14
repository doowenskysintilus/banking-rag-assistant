"""Per-owner BM25 retriever, built from that owner's current document chunks and
cached briefly — rebuilding a `BM25Retriever` means re-tokenizing the whole corpus,
which isn't free, but a stale-for-a-few-minutes lexical index is a much smaller
problem than one that's rebuilt (and DB-hit) on every single query.

Owner-scoping happens at the *corpus fetch* (`fetch_owner_documents`), not by
filtering afterward — another tenant's text never enters this index at all.
"""

from __future__ import annotations

import threading
import time

from langchain_community.retrievers import BM25Retriever
from sqlalchemy.ext.asyncio import AsyncSession

from backend.retrieval.enterprise.documents import fetch_owner_documents

_DEFAULT_TTL_SECONDS = 300

_cache: dict[str, tuple[float, BM25Retriever]] = {}
_cache_lock = threading.Lock()


async def get_owner_bm25_retriever(
    db: AsyncSession, owner_id: str, top_k: int, ttl_seconds: int = _DEFAULT_TTL_SECONDS
) -> BM25Retriever | None:
    """Returns None if the owner has no indexed chunks yet (nothing to search)."""
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(owner_id)
        if cached is not None and (now - cached[0]) < ttl_seconds:
            return cached[1]

    documents = await fetch_owner_documents(db, owner_id)
    if not documents:
        with _cache_lock:
            _cache.pop(owner_id, None)
        return None

    retriever = BM25Retriever.from_documents(documents, k=top_k)

    with _cache_lock:
        _cache[owner_id] = (now, retriever)
    return retriever


def invalidate_owner_bm25_cache(owner_id: str) -> None:
    """Call after ingesting/deleting documents for `owner_id` so the next query
    doesn't have to wait out the TTL to see the change.
    """
    with _cache_lock:
        _cache.pop(owner_id, None)


def clear_bm25_cache() -> None:
    """Test/ops escape hatch — drops every cached owner corpus."""
    with _cache_lock:
        _cache.clear()
