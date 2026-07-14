"""RAGAS-backed metric provider — covers `faithfulness`, `context_precision`,
and `context_recall`: RAGAS's three flagship metrics that are purely
LLM-judged (no separate embedding model needed, unlike its `answer_relevancy`,
which compares embedding similarity between the question and LLM-generated
candidate questions). `answer_relevancy` and `hallucination_rate` are computed
by DeepEval instead (see `deepeval_provider.py`) — this keeps the RAGAS
integration to a single judge-LLM, with no embeddings adapter to build/maintain.

Everything RAGAS/LangChain-specific is imported lazily, inside methods — this
module must stay importable (for `selector.probe_availability`'s trial-import)
even in environments where RAGAS or one of its transitive dependencies isn't
installed or is broken. That is not a hypothetical: at the time of writing,
`ragas` (every version through 0.4.x) unconditionally imports
`langchain_community.chat_models.vertexai`, a module `langchain-community`
removed as of its 0.4.0 "sunsetting" restructuring — so `ragas` fails to import
at all against this project's pinned `langchain-community>=0.4.0`. That is
exactly the kind of transitive breakage `probe_availability`'s real trial-import
(not just `importlib.util.find_spec`) is meant to catch.
"""

from __future__ import annotations

import asyncio
import math

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.evaluation.providers.base import EvaluationProvider
from backend.evaluation.types import EvalSample

logger = get_logger(__name__)

_METRIC_NAMES = ("faithfulness", "context_precision", "context_recall")


class RagasEvaluationProvider(EvaluationProvider):
    name = "ragas"
    supported_metrics = _METRIC_NAMES

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._availability_cache: bool | None = None

    def _build_llm(self):
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        chat = ChatOpenAI(
            model=self._settings.openai_chat_model,
            api_key=self._settings.openai_api_key or None,
            max_tokens=self._settings.openai_chat_max_tokens,
        )
        return LangchainLLMWrapper(chat)

    def _build_metrics(self, requested: list[str], llm) -> list:
        from ragas.metrics import context_precision, context_recall, faithfulness

        registry = {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        selected = []
        for name in requested:
            metric = registry.get(name)
            if metric is None:
                continue
            metric.llm = llm
            selected.append(metric)
        return selected

    def _evaluate_sync(self, samples: list[EvalSample], requested: list[str]):
        from ragas import EvaluationDataset, SingleTurnSample, evaluate

        llm = self._build_llm()
        ragas_metrics = self._build_metrics(requested, llm)

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=s.question,
                    response=s.answer,
                    retrieved_contexts=s.contexts or [""],
                    reference=s.ground_truth,
                )
                for s in samples
            ]
        )

        return evaluate(
            dataset,
            metrics=ragas_metrics,
            llm=llm,
            raise_exceptions=False,
            show_progress=False,
        )

    async def evaluate(
        self, samples: list[EvalSample], metrics: list[str]
    ) -> dict[str, list[float | None]]:
        requested = [m for m in metrics if m in _METRIC_NAMES]
        if not requested or not samples:
            return {m: [None] * len(samples) for m in requested}

        # ragas.evaluate() is a blocking call that drives its own internal event
        # loop (via ragas.executor.Executor) — running it in a worker thread
        # avoids "asyncio.run() cannot be called from a running event loop"
        # when this coroutine is itself awaited from inside FastAPI's loop.
        result = await asyncio.to_thread(self._evaluate_sync, samples, requested)

        scores: dict[str, list[float | None]] = {name: [] for name in requested}
        for row in result.scores:
            for name in requested:
                value = row.get(name)
                is_nan = isinstance(value, float) and math.isnan(value)
                scores[name].append(None if value is None or is_nan else float(value))
        return scores

    @property
    def is_available(self) -> bool:
        # Cached after the first check: `ragas.executor` calls
        # `nest_asyncio.apply()` unconditionally at import time — *before* the
        # module can even fail (see the module docstring) — which is a
        # process-wide side effect on the asyncio event loop. Repeating this
        # trial-import on every call (e.g. once per evaluation run) would
        # re-trigger it every time for no benefit, since availability can't
        # change within a running process.
        if self._availability_cache is None:
            self._availability_cache = self._probe_availability()
        return self._availability_cache

    @staticmethod
    def _probe_availability() -> bool:
        try:
            import ragas  # noqa: F401
            import ragas.metrics  # noqa: F401
            from langchain_openai import ChatOpenAI  # noqa: F401
            from ragas.llms import LangchainLLMWrapper  # noqa: F401
        except Exception:  # noqa: BLE001 - any transitive import failure means "unavailable"
            return False
        return True
