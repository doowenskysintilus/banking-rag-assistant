"""DeepEval-backed metric provider — covers `answer_relevancy` and
`hallucination_rate`. RAGAS covers the other three (see `ragas_provider.py`);
this split keeps each framework to metrics it's a natural fit for and avoids
needing a RAGAS embeddings adapter (RAGAS's own `answer_relevancy` needs one;
DeepEval's doesn't — it's LLM-judged like everything else here).

DeepEval actually defaults to OpenAI's own `GPTModel` already when given no
custom judge — but that reads `OPENAI_API_KEY` straight from the process
environment, bypassing this app's `Settings`. `_build_openai_judge` below
builds an equivalent judge explicitly from `Settings.openai_api_key`/
`openai_chat_model` instead, so it stays configured the same way as every
other OpenAI call in this app rather than depending on an ambient env var
DeepEval happens to read on its own.

`hallucination_rate` is DeepEval's `HallucinationMetric`, which actually scores
the *hallucination fraction* (0 = fully grounded, 1 = fully hallucinated) — the
inverse of what this framework's `MetricScore.score` is defined to mean
(higher is always better, see `backend.evaluation.types`). This provider
returns `1 - raw_score` so every metric follows that same convention.

Everything DeepEval-specific is imported lazily, inside methods, for the same
reason as `ragas_provider.py`: this module must stay importable even where
DeepEval isn't installed, so `selector.probe_availability`'s trial-import can
report unavailability cleanly instead of the whole app failing to start.
"""

from __future__ import annotations

import asyncio

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.evaluation.providers.base import EvaluationProvider
from backend.evaluation.types import EvalSample

logger = get_logger(__name__)

_METRIC_NAMES = ("answer_relevancy", "hallucination_rate")


def _build_openai_judge(settings: Settings):
    """A neutral, no-persona OpenAI call used purely as DeepEval's LLM-as-judge.

    Deliberately not `backend.llm.client.OpenAIClient`: that client always
    injects the banking-assistant `SYSTEM_PROMPT`, which is the wrong persona
    for judging relevancy/hallucination — the judge must answer DeepEval's own
    grading instructions, not roleplay a bank support agent.
    """
    from deepeval.models.base_model import DeepEvalBaseLLM
    from openai import AsyncOpenAI

    class OpenAIJudgeLLM(DeepEvalBaseLLM):
        def __init__(self) -> None:
            self._client: AsyncOpenAI | None = None
            self._model = settings.openai_chat_model
            self._max_tokens = settings.openai_chat_max_tokens
            super().__init__(settings.openai_chat_model)

        def load_model(self):
            # Deliberately not constructing AsyncOpenAI here: unlike the
            # previous Anthropic client, `AsyncOpenAI(api_key=None)` raises
            # immediately (rather than failing lazily on first call) when no
            # key is resolvable from either the constructor or the
            # OPENAI_API_KEY env var. Deferring construction to a_generate
            # keeps this judge constructible even when no key is configured,
            # as long as it's never actually asked to judge anything.
            return self._model

        def get_model_name(self) -> str:
            return self._model

        async def a_generate(self, prompt: str, **kwargs) -> str:
            if self._client is None:
                self._client = AsyncOpenAI(api_key=settings.openai_api_key or None)
            response = await self._client.chat.completions.create(
                model=self._model,
                max_completion_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        def generate(self, prompt: str, **kwargs) -> str:
            return asyncio.run(self.a_generate(prompt))

    return OpenAIJudgeLLM()


class DeepEvalEvaluationProvider(EvaluationProvider):
    name = "deepeval"
    supported_metrics = _METRIC_NAMES

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._availability_cache: bool | None = None

    def _build_metric(self, metric_name: str, judge_llm):
        from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric

        if metric_name == "answer_relevancy":
            return AnswerRelevancyMetric(
                model=judge_llm, include_reason=True, async_mode=True
            )
        if metric_name == "hallucination_rate":
            return HallucinationMetric(
                model=judge_llm, include_reason=True, async_mode=True
            )
        raise ValueError(f"Unsupported metric for DeepEval provider: {metric_name}")

    async def _score_one(
        self, metric_name: str, metric, sample: EvalSample
    ) -> float | None:
        from deepeval.test_case import LLMTestCase

        try:
            if metric_name == "hallucination_rate":
                test_case = LLMTestCase(
                    input=sample.question,
                    actual_output=sample.answer,
                    context=sample.contexts or [""],
                )
            else:
                test_case = LLMTestCase(
                    input=sample.question, actual_output=sample.answer
                )

            await metric.a_measure(test_case, _show_indicator=False)
            raw_score = float(metric.score)
        except Exception:  # noqa: BLE001 - one judge-call failure must not sink the whole run
            logger.exception(
                "deepeval_metric_scoring_failed",
                extra={"metric": metric_name, "sample_id": sample.sample_id},
            )
            return None

        # HallucinationMetric's raw score is a hallucination *fraction* (0 =
        # grounded, 1 = fully hallucinated) — inverted here so every metric this
        # framework returns follows the same "higher is better" convention.
        return 1.0 - raw_score if metric_name == "hallucination_rate" else raw_score

    async def evaluate(
        self, samples: list[EvalSample], metrics: list[str]
    ) -> dict[str, list[float | None]]:
        requested = [m for m in metrics if m in _METRIC_NAMES]
        scores: dict[str, list[float | None]] = {name: [] for name in requested}
        if not requested or not samples:
            return scores

        judge_llm = _build_openai_judge(self._settings)
        for metric_name in requested:
            metric = self._build_metric(metric_name, judge_llm)
            for sample in samples:
                scores[metric_name].append(
                    await self._score_one(metric_name, metric, sample)
                )
        return scores

    @property
    def is_available(self) -> bool:
        # Cached after the first check — see the matching note on
        # RagasEvaluationProvider.is_available; availability can't change
        # within a running process, so there's no reason to re-import on
        # every evaluation run.
        if self._availability_cache is None:
            self._availability_cache = self._probe_availability()
        return self._availability_cache

    @staticmethod
    def _probe_availability() -> bool:
        try:
            import openai  # noqa: F401
            from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric  # noqa: F401
            from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: F401
            from deepeval.test_case import LLMTestCase  # noqa: F401
        except Exception:  # noqa: BLE001 - any transitive import failure means "unavailable"
            return False
        return True
