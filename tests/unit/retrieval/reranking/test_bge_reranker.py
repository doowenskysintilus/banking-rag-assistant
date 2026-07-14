import sys
import types


from backend.core.config import Settings
from backend.retrieval.reranking.bge_reranker import BGERerankerProvider


def _settings(**overrides) -> Settings:
    return Settings(secret_key="x" * 16, **overrides)


def _install_fake_flag_embedding(monkeypatch, compute_score_fn):
    fake_module = types.ModuleType("FlagEmbedding")

    class FakeFlagReranker:
        def __init__(self, model_name, use_fp16=True):
            self.model_name = model_name

        def compute_score(self, pairs, normalize=True):
            return compute_score_fn(pairs, normalize)

    fake_module.FlagReranker = FakeFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)


def _install_fake_cross_encoder(monkeypatch, predict_fn):
    fake_module = types.ModuleType("sentence_transformers")

    class FakeCrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            return predict_fn(pairs)

    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


class TestFlagEmbeddingBackend:
    async def test_uses_flag_embedding_when_available(self, monkeypatch):
        captured = {}

        def compute_score_fn(pairs, normalize):
            captured["pairs"] = pairs
            captured["normalize"] = normalize
            return [0.9, 0.1]

        _install_fake_flag_embedding(monkeypatch, compute_score_fn)

        provider = BGERerankerProvider(
            _settings(bge_reranker_model="BAAI/bge-reranker-v2-m3")
        )
        scores = await provider.score("query", ["doc a", "doc b"])

        assert scores == [0.9, 0.1]
        assert captured["normalize"] is True
        assert captured["pairs"] == [["query", "doc a"], ["query", "doc b"]]

    async def test_handles_a_single_float_return_from_compute_score(self, monkeypatch):
        _install_fake_flag_embedding(monkeypatch, lambda pairs, normalize: 0.7)

        provider = BGERerankerProvider(_settings())
        scores = await provider.score("query", ["only one document"])

        assert scores == [0.7]

    async def test_empty_documents_returns_empty_without_loading_model(
        self, monkeypatch
    ):
        load_count = {"n": 0}

        def compute_score_fn(pairs, normalize):
            load_count["n"] += 1
            return []

        _install_fake_flag_embedding(monkeypatch, compute_score_fn)
        provider = BGERerankerProvider(_settings())

        assert await provider.score("query", []) == []
        assert load_count["n"] == 0


class TestCrossEncoderFallback:
    async def test_falls_back_to_cross_encoder_when_flag_embedding_missing(
        self, monkeypatch
    ):
        # FlagEmbedding genuinely isn't installed in the test environment, so
        # removing any leftover fake from sys.modules is enough to force the
        # real ImportError → CrossEncoder fallback path, with no need to patch
        # builtins.__import__.
        monkeypatch.delitem(sys.modules, "FlagEmbedding", raising=False)
        _install_fake_cross_encoder(monkeypatch, lambda pairs: [2.0, -2.0])

        provider = BGERerankerProvider(_settings())
        scores = await provider.score("query", ["relevant doc", "irrelevant doc"])

        # sigmoid(2.0) > 0.5 > sigmoid(-2.0)
        assert scores[0] > 0.5
        assert scores[1] < 0.5

    async def test_model_loaded_lazily_only_once(self, monkeypatch):
        load_count = {"n": 0}

        def compute_score_fn(pairs, normalize):
            load_count["n"] += 1
            return [0.5 for _ in pairs]

        _install_fake_flag_embedding(monkeypatch, compute_score_fn)
        provider = BGERerankerProvider(_settings())

        await provider.score("q", ["a", "b"])
        await provider.score("q", ["c"])

        assert load_count["n"] == 2  # model loaded once, scored twice


class TestIsAvailable:
    def test_true_when_flag_embedding_importable(self, monkeypatch):
        _install_fake_flag_embedding(monkeypatch, lambda pairs, normalize: [])
        assert BGERerankerProvider(_settings()).is_available is True

    def test_true_when_only_sentence_transformers_importable(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "FlagEmbedding", raising=False)
        _install_fake_cross_encoder(monkeypatch, lambda pairs: [])
        assert BGERerankerProvider(_settings()).is_available is True
