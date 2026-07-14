"""Abstract provider interface — one implementation per evaluation framework
(RAGAS, DeepEval). Mirrors the Strategy pattern already used for embeddings/
vector-store/reranker providers elsewhere in this app, with one difference:
those pick a single active implementation, while here `EvaluationService`
composes several providers together, since RAGAS and DeepEval are each used for
a *different subset* of the five required metrics (see selector.py) rather than
being interchangeable alternatives for the same one.
"""

from __future__ import annotations

import abc

from backend.evaluation.types import EvalSample


class EvaluationProvider(abc.ABC):
    name: str
    supported_metrics: tuple[str, ...]

    @abc.abstractmethod
    async def evaluate(
        self, samples: list[EvalSample], metrics: list[str]
    ) -> dict[str, list[float | None]]:
        """Returns `{metric_name: [per-sample score, in the same order as
        `samples`]}` for each of `metrics` this provider was asked to compute
        (a subset of `supported_metrics`). A `None` entry means that one sample
        could not be scored (e.g. a judge-LLM call failed) — callers must treat
        that as missing data, not as a zero score.
        """

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this provider's dependencies (and any required transitive
        imports) actually work right now — see `probe_availability` in
        selector.py for why this must be a real trial-import, not just an
        `importlib.util.find_spec` check.
        """
