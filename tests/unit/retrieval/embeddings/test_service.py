import numpy as np
import pytest

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError
from backend.retrieval.embeddings.cache import (
    InMemoryEmbeddingCache,
    NullEmbeddingCache,
)
from backend.retrieval.embeddings.service import EmbeddingService
from tests.unit.retrieval.embeddings.fakes import FakeProvider


def _settings(**overrides) -> Settings:
    return Settings(secret_key="x" * 16, **overrides)


def _service(fake_provider: FakeProvider, **overrides) -> EmbeddingService:
    settings = _settings(embedding_dim=fake_provider.dimension, **overrides)
    return EmbeddingService(
        settings, provider=fake_provider, cache=NullEmbeddingCache()
    )


class TestDimensionValidation:
    def test_raises_when_provider_dimension_does_not_match_settings(
        self, fake_provider
    ):
        settings = _settings(embedding_dim=fake_provider.dimension + 1)
        with pytest.raises(ExternalServiceError):
            EmbeddingService(
                settings, provider=fake_provider, cache=NullEmbeddingCache()
            )

    def test_succeeds_when_dimensions_match(self, fake_provider):
        service = _service(fake_provider)
        assert service.dimension == fake_provider.dimension
        assert service.provider_name == "fake"


class TestEmbedTexts:
    async def test_returns_empty_array_for_empty_input(self, fake_provider):
        service = _service(fake_provider)
        result = await service.embed_texts([])
        assert result.shape == (0, fake_provider.dimension)
        assert fake_provider.calls == []

    async def test_embeds_all_texts_as_documents(self, fake_provider):
        service = _service(fake_provider)
        result = await service.embed_texts(["alpha", "beta"])
        assert result.shape == (2, fake_provider.dimension)
        assert fake_provider.is_query_flags == [False]

    async def test_embed_query_uses_the_query_flag(self, fake_provider):
        service = _service(fake_provider)
        await service.embed_query("what is the limit?")
        assert fake_provider.is_query_flags == [True]

    async def test_embed_query_returns_a_single_vector_not_a_batch(self, fake_provider):
        service = _service(fake_provider)
        result = await service.embed_query("hello")
        assert result.shape == (fake_provider.dimension,)


class TestBatching:
    async def test_splits_large_input_into_configured_batch_size(self, fake_provider):
        service = _service(fake_provider, embedding_batch_size=3)
        texts = [f"text-{i}" for i in range(7)]
        await service.embed_texts(texts)
        batch_sizes = [len(call) for call in fake_provider.calls]
        assert batch_sizes == [3, 3, 1]

    async def test_result_order_matches_input_order_across_batches(self, fake_provider):
        service = _service(fake_provider, embedding_batch_size=2)
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        result = await service.embed_texts(texts)
        # FakeProvider encodes len(text) into column 0.
        assert [int(v) for v in result[:, 0]] == [1, 2, 3, 4, 5]


class TestDeduplication:
    async def test_identical_texts_in_one_call_are_embedded_only_once(
        self, fake_provider
    ):
        service = _service(fake_provider)
        result = await service.embed_texts(["repeat", "other", "repeat"])

        total_embedded = sum(len(call) for call in fake_provider.calls)
        assert total_embedded == 2  # "repeat" embedded once, "other" once
        np.testing.assert_array_equal(result[0], result[2])  # both "repeat" rows match


class TestCaching:
    async def test_second_call_with_same_text_hits_the_cache(self, fake_provider):
        settings = _settings(embedding_dim=fake_provider.dimension)
        service = EmbeddingService(
            settings, provider=fake_provider, cache=InMemoryEmbeddingCache()
        )

        first = await service.embed_texts(["cached text"])
        second = await service.embed_texts(["cached text"])

        assert len(fake_provider.calls) == 1  # only embedded once across both calls
        np.testing.assert_array_equal(first, second)

    async def test_cache_key_is_provider_specific(self, fake_provider):
        cache = InMemoryEmbeddingCache()
        settings = _settings(embedding_dim=fake_provider.dimension)
        service = EmbeddingService(settings, provider=fake_provider, cache=cache)
        await service.embed_texts(["hello"])

        other_provider = FakeProvider(dimension=fake_provider.dimension)
        other_provider.name = "other-fake"
        other_service = EmbeddingService(settings, provider=other_provider, cache=cache)
        await other_service.embed_texts(["hello"])

        # different provider -> different cache key -> both actually embedded
        assert len(fake_provider.calls) == 1
        assert len(other_provider.calls) == 1

    async def test_query_and_document_cache_entries_are_independent(
        self, fake_provider
    ):
        # Same text embedded once as a document and once as a query must not
        # collide in the cache (they can be genuinely different vectors for
        # asymmetric providers like BGE/Instructor-XL).
        cache = InMemoryEmbeddingCache()
        settings = _settings(embedding_dim=fake_provider.dimension)
        service = EmbeddingService(settings, provider=fake_provider, cache=cache)

        await service.embed_texts(["same text"])
        await service.embed_query("same text")

        assert len(fake_provider.calls) == 2
        assert fake_provider.is_query_flags == [False, True]
