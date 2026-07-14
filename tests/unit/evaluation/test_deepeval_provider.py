"""Tests for DeepEvalEvaluationProvider. DeepEval and `openai` are both
really importable in this environment (unlike ragas — see
test_ragas_provider.py), so `is_available` is exercised directly for real. For
`evaluate()`, the actual `AnswerRelevancyMetric`/`HallucinationMetric` classes
are monkeypatched with scripted fakes that never call a judge LLM — no network
call, no API key, fully deterministic — while still exercising this provider's
own real logic (score inversion for hallucination_rate, exception → None).
"""

from __future__ import annotations

import deepeval.metrics as deepeval_metrics_module

from backend.core.config import Settings
from backend.evaluation.providers.deepeval_provider import DeepEvalEvaluationProvider
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


class _ScriptedMetric:
    """Fake DeepEval metric: returns scripted scores in call order, never
    touches a real judge LLM.
    """

    def __init__(self, scores: list[float | Exception], **kwargs) -> None:
        self._scores = list(scores)
        self.score: float | None = None

    async def a_measure(self, test_case, _show_indicator=True):
        self.last_test_case = test_case
        value = self._scores.pop(0)
        if isinstance(value, Exception):
            raise value
        self.score = value


class TestIsAvailable:
    def test_true_against_the_real_installed_environment(self):
        assert DeepEvalEvaluationProvider(_settings()).is_available is True

    def test_cached_after_first_call(self, monkeypatch):
        provider = DeepEvalEvaluationProvider(_settings())
        calls = {"n": 0}

        def _probe():
            calls["n"] += 1
            return True

        monkeypatch.setattr(provider, "_probe_availability", _probe)
        assert provider.is_available is True
        assert provider.is_available is True
        assert calls["n"] == 1


class TestEvaluate:
    async def test_answer_relevancy_scores_pass_through_unmodified(self, monkeypatch):
        scripted = _ScriptedMetric(scores=[0.9, 0.6])
        monkeypatch.setattr(
            deepeval_metrics_module, "AnswerRelevancyMetric", lambda **kwargs: scripted
        )

        provider = DeepEvalEvaluationProvider(_settings())
        result = await provider.evaluate(_samples(2), ["answer_relevancy"])

        assert result["answer_relevancy"] == [0.9, 0.6]

    async def test_hallucination_rate_is_inverted_from_the_raw_fraction(
        self, monkeypatch
    ):
        # raw hallucination fraction 0.2 -> non-hallucination score 0.8
        scripted = _ScriptedMetric(scores=[0.2])
        monkeypatch.setattr(
            deepeval_metrics_module, "HallucinationMetric", lambda **kwargs: scripted
        )

        provider = DeepEvalEvaluationProvider(_settings())
        result = await provider.evaluate(_samples(1), ["hallucination_rate"])

        assert result["hallucination_rate"] == [0.8]

    async def test_hallucination_test_case_uses_context_field_not_retrieval_context(
        self, monkeypatch
    ):
        scripted = _ScriptedMetric(scores=[0.0])
        monkeypatch.setattr(
            deepeval_metrics_module, "HallucinationMetric", lambda **kwargs: scripted
        )

        provider = DeepEvalEvaluationProvider(_settings())
        samples = [
            EvalSample(
                sample_id="s0",
                question="q",
                answer="a",
                contexts=["ctx-1", "ctx-2"],
                ground_truth="gt",
            )
        ]
        await provider.evaluate(samples, ["hallucination_rate"])

        assert scripted.last_test_case.context == ["ctx-1", "ctx-2"]

    async def test_a_failed_judge_call_yields_none_for_that_sample_without_stopping_the_batch(
        self, monkeypatch
    ):
        scripted = _ScriptedMetric(scores=[0.9, RuntimeError("judge call failed"), 0.7])
        monkeypatch.setattr(
            deepeval_metrics_module, "AnswerRelevancyMetric", lambda **kwargs: scripted
        )

        provider = DeepEvalEvaluationProvider(_settings())
        result = await provider.evaluate(_samples(3), ["answer_relevancy"])

        assert result["answer_relevancy"] == [0.9, None, 0.7]

    async def test_ignores_metrics_it_does_not_support(self, monkeypatch):
        scripted = _ScriptedMetric(scores=[0.9])
        monkeypatch.setattr(
            deepeval_metrics_module, "AnswerRelevancyMetric", lambda **kwargs: scripted
        )

        provider = DeepEvalEvaluationProvider(_settings())
        result = await provider.evaluate(
            _samples(1), ["answer_relevancy", "faithfulness"]
        )

        assert "faithfulness" not in result
        assert result["answer_relevancy"] == [0.9]

    async def test_empty_samples_returns_empty_lists_without_building_a_judge(
        self, monkeypatch
    ):
        def _boom(settings):
            raise AssertionError(
                "should not build a judge LLM when there are no samples"
            )

        monkeypatch.setattr(
            "backend.evaluation.providers.deepeval_provider._build_openai_judge", _boom
        )

        provider = DeepEvalEvaluationProvider(_settings())
        result = await provider.evaluate([], ["answer_relevancy"])

        assert result == {"answer_relevancy": []}
