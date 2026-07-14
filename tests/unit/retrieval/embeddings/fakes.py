"""Fake dependency-free stand-ins shared across embeddings module tests."""

from __future__ import annotations

import numpy as np

from backend.retrieval.embeddings.types import EmbeddingProvider


class FakeProvider(EmbeddingProvider):
    """Deterministic, dependency-free stand-in for a real provider. Encodes text
    length into the vector so tests can assert on which texts were actually
    embedded (via `.calls`) without needing real models.
    """

    name = "fake"

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []
        self.is_query_flags: list[bool] = []
        self._available = True

    async def embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        self.calls.append(list(texts))
        self.is_query_flags.append(is_query)
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for i, text in enumerate(texts):
            vectors[i, 0] = len(text)
            vectors[i, 1] = 1.0 if is_query else 0.0
        return vectors

    @property
    def is_available(self) -> bool:
        return self._available


class FakeRedisPipeline:
    def __init__(self, store: dict[str, bytes], fail: bool) -> None:
        self._store = store
        self._fail = fail
        self._ops: list[tuple[str, bytes]] = []

    def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        self._ops.append((key, value))

    async def execute(self) -> None:
        if self._fail:
            raise ConnectionError("fake redis is down")
        for key, value in self._ops:
            self._store[key] = value


class FakeRedisClient:
    """Minimal stand-in for `redis.asyncio.Redis` — enough surface for
    `RedisEmbeddingCache` (mget + pipeline().set()/.execute()).
    """

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, bytes] = {}
        self.fail = fail

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        if self.fail:
            raise ConnectionError("fake redis is down")
        return [self.store.get(key) for key in keys]

    def pipeline(self, transaction: bool = False) -> FakeRedisPipeline:
        return FakeRedisPipeline(self.store, self.fail)
