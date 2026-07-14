import sys
import types

from backend.core.config import Settings
from backend.retrieval.reranking.cross_encoder import CrossEncoderRerankerProvider


def _settings(**overrides) -> Settings:
    return Settings(secret_key="x" * 16, **overrides)


def _install_fake_cross_encoder(monkeypatch, predict_fn):
    fake_module = types.ModuleType("sentence_transformers")

    class FakeCrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            return predict_fn(pairs)

    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


class TestScore:
    async def test_scores_are_sigmoid_normalized(self, monkeypatch):
        _install_fake_cross_encoder(monkeypatch, lambda pairs: [5.0, -5.0, 0.0])

        provider = CrossEncoderRerankerProvider(_settings())
        scores = await provider.score("query", ["a", "b", "c"])

        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores[0] > scores[2] > scores[1]

    async def test_passes_query_document_pairs_to_the_model(self, monkeypatch):
        captured = {}

        def predict_fn(pairs):
            captured["pairs"] = pairs
            return [0.0] * len(pairs)

        _install_fake_cross_encoder(monkeypatch, predict_fn)

        provider = CrossEncoderRerankerProvider(
            _settings(cross_encoder_reranker_model="some/model")
        )
        await provider.score("what is the fee?", ["doc one", "doc two"])

        assert captured["pairs"] == [
            ["what is the fee?", "doc one"],
            ["what is the fee?", "doc two"],
        ]

    async def test_empty_documents_returns_empty_without_loading_model(
        self, monkeypatch
    ):
        load_count = {"n": 0}
        fake_module = types.ModuleType("sentence_transformers")

        class FakeCrossEncoder:
            def __init__(self, model_name):
                load_count["n"] += 1

            def predict(self, pairs):
                return []

        fake_module.CrossEncoder = FakeCrossEncoder
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        provider = CrossEncoderRerankerProvider(_settings())
        assert await provider.score("query", []) == []
        assert load_count["n"] == 0

    async def test_model_loaded_lazily_only_once(self, monkeypatch):
        load_count = {"n": 0}
        fake_module = types.ModuleType("sentence_transformers")

        class FakeCrossEncoder:
            def __init__(self, model_name):
                load_count["n"] += 1

            def predict(self, pairs):
                return [0.0 for _ in pairs]

        fake_module.CrossEncoder = FakeCrossEncoder
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        provider = CrossEncoderRerankerProvider(_settings())
        await provider.score("q", ["a"])
        await provider.score("q", ["b"])

        assert load_count["n"] == 1


class TestIsAvailable:
    def test_true_when_sentence_transformers_importable(self, monkeypatch):
        _install_fake_cross_encoder(monkeypatch, lambda pairs: [])
        assert CrossEncoderRerankerProvider(_settings()).is_available is True
