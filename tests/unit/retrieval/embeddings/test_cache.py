import numpy as np

from backend.retrieval.embeddings.cache import (
    InMemoryEmbeddingCache,
    NullEmbeddingCache,
    RedisEmbeddingCache,
)
from tests.unit.retrieval.embeddings.fakes import FakeRedisClient


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype="float32")


class TestNullEmbeddingCache:
    async def test_get_many_always_misses(self):
        cache = NullEmbeddingCache()
        assert await cache.get_many(["a", "b"]) == {}

    async def test_set_many_is_a_no_op(self):
        cache = NullEmbeddingCache()
        await cache.set_many({"a": _vec(1.0)})
        assert await cache.get_many(["a"]) == {}


class TestInMemoryEmbeddingCache:
    async def test_set_then_get_roundtrips(self):
        cache = InMemoryEmbeddingCache()
        await cache.set_many({"a": _vec(1.0, 2.0)})
        result = await cache.get_many(["a", "missing"])
        assert set(result.keys()) == {"a"}
        np.testing.assert_array_equal(result["a"], _vec(1.0, 2.0))

    async def test_evicts_least_recently_used_when_over_capacity(self):
        cache = InMemoryEmbeddingCache(max_entries=2)
        await cache.set_many({"a": _vec(1.0)})
        await cache.set_many({"b": _vec(2.0)})
        await cache.set_many({"c": _vec(3.0)})  # evicts "a" (never re-accessed)

        result = await cache.get_many(["a", "b", "c"])
        assert set(result.keys()) == {"b", "c"}

    async def test_get_refreshes_recency(self):
        cache = InMemoryEmbeddingCache(max_entries=2)
        await cache.set_many({"a": _vec(1.0)})
        await cache.set_many({"b": _vec(2.0)})
        await cache.get_many(["a"])  # "a" is now more recently used than "b"
        await cache.set_many({"c": _vec(3.0)})  # evicts "b", not "a"

        result = await cache.get_many(["a", "b", "c"])
        assert set(result.keys()) == {"a", "c"}


class TestRedisEmbeddingCache:
    def _make_cache(
        self, monkeypatch, fail: bool = False
    ) -> tuple[RedisEmbeddingCache, FakeRedisClient]:
        import redis.asyncio as redis_asyncio

        fake_client = FakeRedisClient(fail=fail)
        monkeypatch.setattr(
            redis_asyncio.Redis,
            "from_url",
            classmethod(lambda cls, *a, **kw: fake_client),
        )
        cache = RedisEmbeddingCache("redis://fake", ttl_seconds=60)
        return cache, fake_client

    async def test_set_then_get_roundtrips(self, monkeypatch):
        cache, _ = self._make_cache(monkeypatch)
        await cache.set_many({"a": _vec(1.0, 2.0, 3.0)})
        result = await cache.get_many(["a", "missing"])
        assert set(result.keys()) == {"a"}
        np.testing.assert_array_almost_equal(result["a"], _vec(1.0, 2.0, 3.0))

    async def test_keys_are_prefixed_so_they_cannot_collide_with_other_redis_data(
        self, monkeypatch
    ):
        cache, fake_client = self._make_cache(monkeypatch)
        await cache.set_many({"mykey": _vec(1.0)})
        assert "mykey" not in fake_client.store
        assert any(k.endswith("mykey") for k in fake_client.store)

    async def test_get_many_degrades_to_empty_dict_on_connection_failure(
        self, monkeypatch
    ):
        cache, _ = self._make_cache(monkeypatch, fail=True)
        result = await cache.get_many(["a"])
        assert result == {}

    async def test_set_many_degrades_silently_on_connection_failure(self, monkeypatch):
        cache, _ = self._make_cache(monkeypatch, fail=True)
        # must not raise
        await cache.set_many({"a": _vec(1.0)})

    async def test_empty_keys_short_circuits_without_calling_redis(self, monkeypatch):
        cache, fake_client = self._make_cache(monkeypatch)
        result = await cache.get_many([])
        assert result == {}

    async def test_empty_items_short_circuits_without_calling_redis(self, monkeypatch):
        cache, _ = self._make_cache(monkeypatch)
        await cache.set_many({})  # must not raise
