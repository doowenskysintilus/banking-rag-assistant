"""Shared types for the reranking module."""

from __future__ import annotations

import abc
import math


class RerankerProvider(abc.ABC):
    """One implementation per reranking backend (BGE, generic cross-encoder, ...).

    Scores are always normalized to roughly [0, 1] (a calibrated relevance
    probability, not a raw model logit) so callers can apply a single
    `min_relevance_score` threshold across providers — see
    `backend.retrieval.reranking.service.RerankingService`.
    """

    name: str

    @abc.abstractmethod
    async def score(self, query: str, documents: list[str]) -> list[float]:
        """Returns one relevance score per document, in the same order as
        `documents`. Does not sort or filter — that's the service layer's job.
        """

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this provider's dependencies are importable right now (without
        necessarily having loaded the model yet). Used by the automatic selector.
        """


def sigmoid(x: float) -> float:
    """Maps an unbounded logit to (0, 1). Clamped to avoid overflow on extreme
    (effectively saturated) scores from either direction.
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
