"""Fake, dependency-free `EvaluationProvider` for testing selector/service
orchestration logic without needing RAGAS/DeepEval actually installed or making
real judge-LLM calls.
"""

from __future__ import annotations

from backend.evaluation.providers.base import EvaluationProvider
from backend.evaluation.types import EvalSample


class FakeEvaluationProvider(EvaluationProvider):
    def __init__(
        self,
        name: str,
        supported_metrics: tuple[str, ...],
        *,
        available: bool = True,
        scores: dict[str, list[float | None]] | None = None,
    ) -> None:
        self.name = name
        self.supported_metrics = supported_metrics
        self._available = available
        self._scores = scores or {}
        self.calls: list[tuple[list[EvalSample], list[str]]] = []

    async def evaluate(
        self, samples: list[EvalSample], metrics: list[str]
    ) -> dict[str, list[float | None]]:
        self.calls.append((samples, metrics))
        return {m: self._scores.get(m, [0.5] * len(samples)) for m in metrics}

    @property
    def is_available(self) -> bool:
        return self._available
