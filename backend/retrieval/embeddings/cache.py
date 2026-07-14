"""Embedding cache backends.

Caching is a pure optimization, never a correctness requirement — every
implementation here must degrade to "always a cache miss" rather than raise, so a
Redis outage slows ingestion/search down (recomputing embeddings) instead of
breaking it.
"""

from __future__ import annotations

import abc
import threading
from collections import OrderedDict

import numpy as np

from backend.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache(abc.ABC):
    @abc.abstractmethod
    async def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Returns whatever subset of `keys` is present — missing keys are simply
        absent from the result dict, never an error.
        """

    @abc.abstractmethod
    async def set_many(self, items: dict[str, np.ndarray]) -> None: ...


class NullEmbeddingCache(EmbeddingCache):
    """No-op cache — used when caching is disabled (`EMBEDDING_CACHE_BACKEND=none`)."""

    async def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        return {}

    async def set_many(self, items: dict[str, np.ndarray]) -> None:
        return None


class InMemoryEmbeddingCache(EmbeddingCache):
    """Process-local LRU cache. Useful for local dev/tests without a Redis server,
    or as a same-process complement in front of Redis. Not shared across workers.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()

    async def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        with self._lock:
            for key in keys:
                if key in self._store:
                    self._store.move_to_end(key)
                    result[key] = self._store[key]
        return result

    async def set_many(self, items: dict[str, np.ndarray]) -> None:
        with self._lock:
            for key, vector in items.items():
                self._store[key] = vector
                self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)


class RedisEmbeddingCache(EmbeddingCache):
    """Shared cache across all app instances/workers — the one that actually
    avoids recomputing embeddings for the same text across requests and restarts.

    Vectors are stored as raw float32 bytes (compact, no JSON overhead) under
    `f"{key_prefix}{cache_key}"`, with a TTL since embedding models occasionally
    change and stale-forever entries would silently keep serving an old model's
    vectors.
    """

    def __init__(
        self, redis_url: str, ttl_seconds: int, key_prefix: str = "emb:"
    ) -> None:
        import redis.asyncio as redis

        self._redis = redis.Redis.from_url(redis_url, decode_responses=False)
        self._ttl = ttl_seconds
        self._key_prefix = key_prefix
        self._warned = False

    def _prefixed(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    def _warn_once(self, event: str, exc: Exception) -> None:
        if not self._warned:
            logger.warning(event, extra={"error": str(exc)})
            self._warned = True

    async def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        try:
            raw_values = await self._redis.mget([self._prefixed(k) for k in keys])
        except Exception as exc:  # noqa: BLE001 - cache failure must never break embedding
            self._warn_once("embedding_cache_get_failed", exc)
            return {}

        result: dict[str, np.ndarray] = {}
        for key, raw in zip(keys, raw_values, strict=True):
            if raw is not None:
                result[key] = np.frombuffer(raw, dtype="float32")
        return result

    async def set_many(self, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        try:
            pipeline = self._redis.pipeline(transaction=False)
            for key, vector in items.items():
                pipeline.set(
                    self._prefixed(key),
                    vector.astype("float32").tobytes(),
                    ex=self._ttl,
                )
            await pipeline.execute()
        except Exception as exc:  # noqa: BLE001 - cache failure must never break embedding
            self._warn_once("embedding_cache_set_failed", exc)
