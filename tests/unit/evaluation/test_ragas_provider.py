"""Tests for RagasEvaluationProvider.

`is_available` is tested two ways: against the *real*, unmodified environment
(where, at the time of writing, `ragas` genuinely fails to import — see the
long comment in `backend.evaluation.providers.ragas_provider` — so this
exercises the real defensive trial-import path, not a simulation of it) and
against a fully faked, working `ragas`/`langchain_openai` install (so
`evaluate()`'s score-extraction/NaN-handling logic can be verified
deterministically without either package's real, network-calling behavior).
"""

from __future__ import annotations

import sys
import types

from backend.core.config import Settings
from backend.evaluation.providers.ragas_provider import RagasEvaluationProvider
from backend.evaluation.types import EvalSample


def _settings(**overrides) -> Settings:
    return Settings(secret_key="x" * 16, **overrides)


def _samples(n: int = 2) -> list[EvalSample]:
    return [
        EvalSample(
            sample_id=f"s{i}",
            question=f"q{i}",
            answer=f"a{i}",
            contexts=["ctx"],
            ground_truth="gt",
        )
        for i in range(n)
    ]


def _install_fake_ragas(monkeypatch, *, scores_per_metric: dict[str, list[float]]):
    """Registers minimal fake `ragas`, `ragas.metrics`, `ragas.llms`, and
    `langchain_openai` modules so RagasEvaluationProvider can run its full
    `evaluate()` path against known, deterministic scores.
    """

    class FakeSingleTurnSample:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeEvaluationDataset:
        def __init__(self, samples):
            self.samples = samples

    class FakeMetric:
        def __init__(self, metric_name):
            self._name = metric_name
            self.llm = None

    class FakeEvaluationResult:
        def __init__(self, scores):
            self.scores = scores

    def fake_evaluate(dataset, metrics, llm, raise_exceptions, show_progress):
        names = [m._name for m in metrics]
        rows = []
        for i in range(len(dataset.samples)):
            row = {}
            for name in names:
                values = scores_per_metric.get(name, [])
                row[name] = values[i] if i < len(values) else float("nan")
            rows.append(row)
        return FakeEvaluationResult(rows)

    ragas_module = types.ModuleType("ragas")
    ragas_module.EvaluationDataset = FakeEvaluationDataset
    ragas_module.SingleTurnSample = FakeSingleTurnSample
    ragas_module.evaluate = fake_evaluate

    ragas_metrics_module = types.ModuleType("ragas.metrics")
    ragas_metrics_module.faithfulness = FakeMetric("faithfulness")
    ragas_metrics_module.context_precision = FakeMetric("context_precision")
    ragas_metrics_module.context_recall = FakeMetric("context_recall")
    ragas_module.metrics = ragas_metrics_module

    ragas_llms_module = types.ModuleType("ragas.llms")

    class FakeLangchainLLMWrapper:
        def __init__(self, langchain_llm):
            self.langchain_llm = langchain_llm

    ragas_llms_module.LangchainLLMWrapper = FakeLangchainLLMWrapper
    ragas_module.llms = ragas_llms_module

    langchain_openai_module = types.ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    langchain_openai_module.ChatOpenAI = FakeChatOpenAI

    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", ragas_metrics_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", ragas_llms_module)
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai_module)


class TestIsAvailableAgainstTheRealEnvironment:
    def test_is_available_does_not_raise_and_reports_the_real_state(self):
        # Deliberately not faking anything here: this exercises the real
        # trial-import against whatever is actually installed. The point of
        # this test is that `is_available` degrades to False rather than
        # letting an ImportError/ModuleNotFoundError escape.
        result = RagasEvaluationProvider(_settings()).is_available
        assert isinstance(result, bool)


class TestIsAvailableWithFakeDependencies:
    def test_true_when_ragas_and_langchain_openai_import_cleanly(self, monkeypatch):
        _install_fake_ragas(monkeypatch, scores_per_metric={})
        assert RagasEvaluationProvider(_settings()).is_available is True

    def test_false_when_a_transitive_import_fails(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "langchain_openai", raising=False)

        broken_module = types.ModuleType("langchain_openai")

        def _raise(*args, **kwargs):
            raise ImportError("simulated transitive breakage")

        broken_module.__getattr__ = _raise
        monkeypatch.setitem(sys.modules, "langchain_openai", broken_module)

        # is_available must catch this, not propagate it.
        assert RagasEvaluationProvider(_settings()).is_available is False


class TestEvaluate:
    async def test_returns_scores_in_sample_order_per_metric(self, monkeypatch):
        _install_fake_ragas(
            monkeypatch,
            scores_per_metric={
                "faithfulness": [0.9, 0.6],
                "context_precision": [1.0, 0.5],
            },
        )
        provider = RagasEvaluationProvider(_settings())

        result = await provider.evaluate(
            _samples(2), ["faithfulness", "context_precision"]
        )

        assert result["faithfulness"] == [0.9, 0.6]
        assert result["context_precision"] == [1.0, 0.5]

    async def test_nan_scores_become_none(self, monkeypatch):
        _install_fake_ragas(
            monkeypatch, scores_per_metric={"faithfulness": [0.8, float("nan")]}
        )
        provider = RagasEvaluationProvider(_settings())

        result = await provider.evaluate(_samples(2), ["faithfulness"])

        assert result["faithfulness"] == [0.8, None]

    async def test_ignores_metrics_it_does_not_support(self, monkeypatch):
        _install_fake_ragas(monkeypatch, scores_per_metric={"faithfulness": [0.9, 0.9]})
        provider = RagasEvaluationProvider(_settings())

        result = await provider.evaluate(
            _samples(2), ["faithfulness", "hallucination_rate"]
        )

        assert "hallucination_rate" not in result
        assert result["faithfulness"] == [0.9, 0.9]

    async def test_empty_samples_returns_empty_lists_without_calling_ragas(
        self, monkeypatch
    ):
        provider = RagasEvaluationProvider(_settings())
        result = await provider.evaluate([], ["faithfulness"])
        assert result == {"faithfulness": []}
