import asyncio

import pytest

from backend.retrieval.enterprise.langchain_embeddings import LangChainEmbeddingsAdapter
from tests.unit.retrieval.enterprise.conftest import FakeEmbeddingService


@pytest.fixture
def adapter() -> LangChainEmbeddingsAdapter:
    return LangChainEmbeddingsAdapter(FakeEmbeddingService(dim=4))


class TestAsyncMethods:
    async def test_aembed_documents_returns_plain_lists(self, adapter):
        result = await adapter.aembed_documents(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(vec, list) for vec in result)
        assert len(result[0]) == 4

    async def test_aembed_query_returns_a_single_vector(self, adapter):
        result = await adapter.aembed_query("hello")
        assert isinstance(result, list)
        assert len(result) == 4


class TestSyncMethods:
    def test_sync_methods_work_outside_a_running_event_loop(self, adapter):
        # This test function itself is sync (no running loop), so the defensive
        # asyncio.run() branch should succeed.
        result = adapter.embed_query("hello")
        assert isinstance(result, list)
        assert len(result) == 4

        docs_result = adapter.embed_documents(["a", "b"])
        assert len(docs_result) == 2

    async def test_sync_methods_raise_inside_a_running_event_loop(self, adapter):
        with pytest.raises(RuntimeError):
            adapter.embed_query("hello")
        with pytest.raises(RuntimeError):
            adapter.embed_documents(["a"])


def test_sync_and_async_produce_the_same_vector(adapter):
    sync_result = adapter.embed_query("consistent text")
    async_result = asyncio.run(adapter.aembed_query("consistent text"))
    assert sync_result == async_result
