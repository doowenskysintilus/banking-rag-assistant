import sys
import types

import numpy as np
import pytest

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError, RateLimitedError
from backend.retrieval.embeddings.providers.bge_provider import BGELargeProvider
from backend.retrieval.embeddings.providers.instructor_provider import (
    InstructorXLProvider,
)
from backend.retrieval.embeddings.providers.openai_provider import (
    OpenAIEmbeddingProvider,
)


def _settings(**overrides) -> Settings:
    return Settings(secret_key="x" * 16, **overrides)


class TestOpenAIEmbeddingProvider:
    def test_is_available_reflects_api_key_presence(self):
        assert (
            OpenAIEmbeddingProvider(_settings(openai_api_key="")).is_available is False
        )
        assert (
            OpenAIEmbeddingProvider(_settings(openai_api_key="sk-fake")).is_available
            is True
        )

    async def test_embed_batch_calls_the_api_and_normalizes_vectors(self, monkeypatch):
        import openai

        captured_kwargs = {}

        class FakeEmbeddingItem:
            def __init__(self, embedding):
                self.embedding = embedding

        class FakeResponse:
            def __init__(self, vectors):
                self.data = [FakeEmbeddingItem(v) for v in vectors]

        class FakeEmbeddingsResource:
            async def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeResponse([[3.0, 4.0], [1.0, 0.0]])

        class FakeAsyncOpenAI:
            def __init__(self, api_key=None):
                self.embeddings = FakeEmbeddingsResource()

        monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)

        settings = _settings(openai_api_key="sk-fake", openai_embedding_dim=2)
        provider = OpenAIEmbeddingProvider(settings)

        result = await provider.embed_batch(["hello", "world"], is_query=False)

        assert captured_kwargs["model"] == settings.openai_embedding_model
        assert captured_kwargs["input"] == ["hello", "world"]
        assert result.shape == (2, 2)
        # [3, 4] normalizes to [0.6, 0.8]
        np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)
        np.testing.assert_allclose(result[1], [1.0, 0.0], atol=1e-6)

    async def test_maps_rate_limit_error(self, monkeypatch):
        import openai

        class FakeEmbeddingsResource:
            async def create(self, **kwargs):
                raise openai.RateLimitError(
                    "rate limited", response=_fake_http_response(429), body=None
                )

        class FakeAsyncOpenAI:
            def __init__(self, api_key=None):
                self.embeddings = FakeEmbeddingsResource()

        monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
        provider = OpenAIEmbeddingProvider(_settings(openai_api_key="sk-fake"))

        with pytest.raises(RateLimitedError):
            await provider.embed_batch(["hello"], is_query=False)

    async def test_maps_unexpected_error_to_external_service_error(self, monkeypatch):
        import openai

        class FakeEmbeddingsResource:
            async def create(self, **kwargs):
                raise RuntimeError("boom")

        class FakeAsyncOpenAI:
            def __init__(self, api_key=None):
                self.embeddings = FakeEmbeddingsResource()

        monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
        provider = OpenAIEmbeddingProvider(_settings(openai_api_key="sk-fake"))

        with pytest.raises(ExternalServiceError):
            await provider.embed_batch(["hello"], is_query=False)


def _fake_http_response(status_code: int):
    import httpx

    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return httpx.Response(status_code=status_code, request=request)


class TestBGELargeProvider:
    def _install_fake_sentence_transformers(self, monkeypatch, encode_fn):
        fake_module = types.ModuleType("sentence_transformers")

        class FakeSentenceTransformer:
            def __init__(self, model_name):
                self.model_name = model_name

            def encode(self, inputs, normalize_embeddings=True, convert_to_numpy=True):
                return encode_fn(inputs)

        fake_module.SentenceTransformer = FakeSentenceTransformer
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    async def test_document_text_is_embedded_without_a_prefix(self, monkeypatch):
        captured = {}

        def encode_fn(inputs):
            captured["inputs"] = inputs
            return np.ones((len(inputs), 4), dtype="float32")

        self._install_fake_sentence_transformers(monkeypatch, encode_fn)
        provider = BGELargeProvider(_settings(bge_embedding_dim=4))

        await provider.embed_batch(["overdraft policy"], is_query=False)

        assert captured["inputs"] == ["overdraft policy"]

    async def test_query_text_gets_the_instruction_prefix(self, monkeypatch):
        captured = {}

        def encode_fn(inputs):
            captured["inputs"] = inputs
            return np.ones((len(inputs), 4), dtype="float32")

        self._install_fake_sentence_transformers(monkeypatch, encode_fn)
        provider = BGELargeProvider(_settings(bge_embedding_dim=4))

        await provider.embed_batch(["what is the overdraft limit?"], is_query=True)

        assert captured["inputs"][0].startswith(
            "Represent this sentence for searching relevant passages: "
        )
        assert captured["inputs"][0].endswith("what is the overdraft limit?")

    def test_is_available_true_when_sentence_transformers_is_importable(
        self, monkeypatch
    ):
        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.ModuleType("sentence_transformers"),
        )
        assert BGELargeProvider(_settings()).is_available is True


class TestInstructorXLProvider:
    def _install_fake_instructor(self, monkeypatch, encode_fn):
        fake_module = types.ModuleType("InstructorEmbedding")

        class FakeInstructor:
            def __init__(self, model_name):
                self.model_name = model_name

            def encode(self, pairs):
                return encode_fn(pairs)

        fake_module.INSTRUCTOR = FakeInstructor
        monkeypatch.setitem(sys.modules, "InstructorEmbedding", fake_module)

    async def test_embeds_instruction_text_pairs_with_document_instruction(
        self, monkeypatch
    ):
        captured = {}

        def encode_fn(pairs):
            captured["pairs"] = pairs
            return np.ones((len(pairs), 4), dtype="float32")

        self._install_fake_instructor(monkeypatch, encode_fn)
        settings = _settings(instructor_embedding_dim=4)
        provider = InstructorXLProvider(settings)

        await provider.embed_batch(["loan covenant text"], is_query=False)

        assert captured["pairs"] == [
            [settings.instructor_document_instruction, "loan covenant text"]
        ]

    async def test_embeds_instruction_text_pairs_with_query_instruction(
        self, monkeypatch
    ):
        captured = {}

        def encode_fn(pairs):
            captured["pairs"] = pairs
            return np.ones((len(pairs), 4), dtype="float32")

        self._install_fake_instructor(monkeypatch, encode_fn)
        settings = _settings(instructor_embedding_dim=4)
        provider = InstructorXLProvider(settings)

        await provider.embed_batch(["what is the covenant?"], is_query=True)

        assert captured["pairs"] == [
            [settings.instructor_query_instruction, "what is the covenant?"]
        ]

    async def test_output_is_l2_normalized(self, monkeypatch):
        def encode_fn(pairs):
            return np.array([[3.0, 4.0]], dtype="float32")

        self._install_fake_instructor(monkeypatch, encode_fn)
        provider = InstructorXLProvider(_settings(instructor_embedding_dim=2))

        result = await provider.embed_batch(["text"], is_query=False)
        np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)
