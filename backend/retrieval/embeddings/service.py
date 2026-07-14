"""EmbeddingService — the single entry point the rest of the app depends on.

Ties together: automatic provider selection, batching, bounded parallelism, and an
optional Redis-backed cache. Callers never touch a provider or the cache directly.
"""

from __future__ import annotations

import hashlib

import numpy as np

from backend.core.config import Settings, get_settings
from backend.core.exceptions import ExternalServiceError
from backend.core.logging import get_logger
from backend.retrieval.embeddings.batching import make_batches, run_batches_in_parallel
from backend.retrieval.embeddings.cache import (
    EmbeddingCache,
    InMemoryEmbeddingCache,
    NullEmbeddingCache,
    RedisEmbeddingCache,
)
from backend.retrieval.embeddings.providers.bge_provider import BGELargeProvider
from backend.retrieval.embeddings.providers.instructor_provider import (
    InstructorXLProvider,
)
from backend.retrieval.embeddings.providers.openai_provider import (
    OpenAIEmbeddingProvider,
)
from backend.retrieval.embeddings.selector import (
    probe_availability,
    select_provider_name,
)
from backend.retrieval.embeddings.types import EmbeddingProvider

logger = get_logger(__name__)

_PROVIDER_CLASSES: dict[str, type[EmbeddingProvider]] = {
    "openai": OpenAIEmbeddingProvider,
    "bge_large": BGELargeProvider,
    "instructor_xl": InstructorXLProvider,
}


def build_provider(settings: Settings) -> EmbeddingProvider:
    availability = probe_availability(settings)
    name = select_provider_name(settings, availability)
    provider_cls = _PROVIDER_CLASSES.get(name)
    if provider_cls is None:
        raise ExternalServiceError(f"Unknown embedding provider: {name}")
    return provider_cls(settings)


def build_cache(settings: Settings) -> EmbeddingCache:
    backend = settings.embedding_cache_backend
    if backend == "none":
        return NullEmbeddingCache()
    if backend == "memory":
        return InMemoryEmbeddingCache()
    try:
        return RedisEmbeddingCache(
            settings.redis_url, settings.embedding_cache_ttl_seconds
        )
    except Exception:  # noqa: BLE001 - degrade to an in-process cache, never fail startup
        logger.warning("embedding_cache_redis_init_failed_using_memory_cache")
        return InMemoryEmbeddingCache()


class EmbeddingService:
    def __init__(
        self,
        settings: Settings,
        *,
        provider: EmbeddingProvider | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._provider = provider or build_provider(settings)
        self._cache = cache if cache is not None else build_cache(settings)
        self._batch_size = settings.embedding_batch_size
        self._max_concurrency = settings.embedding_max_concurrency

        if self._provider.dimension != settings.embedding_dim:
            raise ExternalServiceError(
                f"Embedding dimension mismatch: provider '{self._provider.name}' produces "
                f"{self._provider.dimension}-dim vectors but EMBEDDING_DIM={settings.embedding_dim} "
                "(the FAISS index was built for a different dimension). Update EMBEDDING_DIM to "
                "match the active provider, or re-point EMBEDDING_PROVIDER at one that matches "
                "your existing index — then reindex if you actually change providers."
            )

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents/passages for indexing."""
        return await self._embed(texts, is_query=False)

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query."""
        vectors = await self._embed([text], is_query=True)
        return vectors[0]

    async def _embed(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        cache_keys = [self._cache_key(text, is_query=is_query) for text in texts]
        cached = await self._cache.get_many(cache_keys)

        # De-duplicate cache-miss texts before embedding — a batch that repeats the
        # same sentence (common in tables/boilerplate) shouldn't pay for it twice.
        missing_index_of_key: dict[str, int] = {}
        missing_texts: list[str] = []
        for i, key in enumerate(cache_keys):
            if key in cached or key in missing_index_of_key:
                continue
            missing_index_of_key[key] = len(missing_texts)
            missing_texts.append(texts[i])

        if missing_texts:
            batches = make_batches(missing_texts, self._batch_size)
            batch_vectors = await run_batches_in_parallel(
                batches,
                lambda batch: self._provider.embed_batch(batch, is_query=is_query),
                self._max_concurrency,
            )
            computed = np.concatenate(batch_vectors, axis=0)

            newly_cached: dict[str, np.ndarray] = {}
            for key, index in missing_index_of_key.items():
                cached[key] = computed[index]
                newly_cached[key] = computed[index]
            await self._cache.set_many(newly_cached)

        return np.stack([cached[key] for key in cache_keys])

    def _cache_key(self, text: str, *, is_query: bool) -> str:
        # Query vs. document embeddings of the same text can genuinely differ for
        # asymmetric providers (BGE's query instruction prefix, Instructor-XL's
        # per-side instruction) — the flag must be part of the key.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        side = "q" if is_query else "d"
        return f"{self._provider.name}:{side}:{digest}"


_embedding_singleton: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_singleton
    if _embedding_singleton is None:
        _embedding_singleton = EmbeddingService(get_settings())
    return _embedding_singleton
