"""Strategy factory: picks the `VectorStoreBase` implementation to instantiate
based on `settings.vector_store_provider`.

Adding a third backend (Qdrant, Weaviate, ...) later is a two-line change: write
the class, register it in `_STRATEGIES`.
"""

from __future__ import annotations

from backend.core.config import Settings, get_settings
from backend.core.exceptions import ValidationAppError
from backend.retrieval.vector_store.faiss_store import FAISSVectorStore
from backend.retrieval.vector_store.pinecone_store import PineconeVectorStore
from backend.retrieval.vector_store.types import VectorStoreBase

_STRATEGIES: dict[str, type[VectorStoreBase]] = {
    "faiss": FAISSVectorStore,
    "pinecone": PineconeVectorStore,
}


def build_vector_store(settings: Settings) -> VectorStoreBase:
    """Constructs a fresh `VectorStoreBase` strategy instance for `settings`.

    Split out from `get_vector_store()` (the process-wide singleton accessor)
    so callers that need an independent instance — tests, scripts, a future
    reindexing job comparing two providers side by side — don't have to fight
    the module-level cache.
    """
    strategy_cls = _STRATEGIES.get(settings.vector_store_provider)
    if strategy_cls is None:
        raise ValidationAppError(
            f"Unknown vector store provider: {settings.vector_store_provider!r}. "
            f"Supported: {sorted(_STRATEGIES)}"
        )
    return strategy_cls(settings)


_store_singleton: VectorStoreBase | None = None


def get_vector_store() -> VectorStoreBase:
    """FastAPI dependency / general-purpose accessor — returns a process-wide
    singleton so the (potentially expensive-to-construct) index/connection is
    reused across requests.
    """
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = build_vector_store(get_settings())
    return _store_singleton
