"""Evaluation harness: runs a scoring function (a reranker provider's `.score()`,
or a no-op baseline) over a labeled example set and reports standard IR metrics
plus the hallucination-risk proxy `grounding_precision`.
"""

from __future__ import annotations

import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from backend.retrieval.reranking.metrics import (
    average_precision,
    grounding_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

ScoreFn = Callable[[str, list[str]], Awaitable[list[float]]]


@dataclass
class EvaluationExample:
    query: str
    candidate_ids: list[str]
    candidate_texts: list[str]
    relevant_ids: set[str]


@dataclass
class EvaluationReport:
    provider_name: str
    num_examples: int
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    map_score: float
    ndcg_at_k: float
    grounding_precision: float


async def _baseline_score_fn(query: str, documents: list[str]) -> list[float]:
    """No reranking: preserves original retrieval order via decreasing placeholder
    scores. These aren't calibrated relevance probabilities (there's no model
    behind them), so this baseline's `grounding_precision` isn't a meaningful
    hallucination-risk signal — only a real reranker's confidence is.
    """
    n = len(documents)
    if n == 0:
        return []
    return [1.0 - (i / n) for i in range(n)]


class RerankEvaluator:
    def __init__(self, k: int = 5, relevance_threshold: float = 0.5) -> None:
        self._k = k
        self._threshold = relevance_threshold

    async def evaluate(
        self,
        examples: list[EvaluationExample],
        score_fn: ScoreFn | None = None,
        provider_name: str = "baseline",
    ) -> EvaluationReport:
        fn = score_fn or _baseline_score_fn

        precisions, recalls, reciprocal_ranks, average_precisions, ndcgs, groundings = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for example in examples:
            scores = await fn(example.query, example.candidate_texts)
            ranked = sorted(
                zip(example.candidate_ids, scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
            ranked_ids = [doc_id for doc_id, _ in ranked]
            ranked_scores = [score for _, score in ranked]

            precisions.append(precision_at_k(ranked_ids, example.relevant_ids, self._k))
            recalls.append(recall_at_k(ranked_ids, example.relevant_ids, self._k))
            reciprocal_ranks.append(reciprocal_rank(ranked_ids, example.relevant_ids))
            average_precisions.append(
                average_precision(ranked_ids, example.relevant_ids)
            )
            ndcgs.append(ndcg_at_k(ranked_ids, example.relevant_ids, self._k))
            groundings.append(
                grounding_precision(ranked_scores, self._threshold, self._k)
            )

        return EvaluationReport(
            provider_name=provider_name,
            num_examples=len(examples),
            k=self._k,
            precision_at_k=statistics.mean(precisions) if precisions else 0.0,
            recall_at_k=statistics.mean(recalls) if recalls else 0.0,
            mrr=statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            map_score=statistics.mean(average_precisions)
            if average_precisions
            else 0.0,
            ndcg_at_k=statistics.mean(ndcgs) if ndcgs else 0.0,
            grounding_precision=statistics.mean(groundings) if groundings else 0.0,
        )
