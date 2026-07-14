"""Standard information-retrieval evaluation metrics, plus one metric specific to
this system's "reduce hallucinations" goal (`grounding_precision`).

All functions are pure and take plain `ranked_ids`/`relevant_ids` (or scores) so
they can score a reranker's output, a raw retriever's output, or a hand-written
baseline identically — see `backend.retrieval.reranking.evaluation`.
"""

from __future__ import annotations

import math


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(top_k)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / position
    return 0.0


def average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean of precision@i evaluated at each rank i where a relevant document
    appears — the per-query term that gets averaged into MAP across a query set.
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / position
    return precision_sum / len(relevant_ids)


def dcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for position, doc_id in enumerate(ranked_ids[:k], start=1):
        relevance = 1.0 if doc_id in relevant_ids else 0.0
        dcg += relevance / math.log2(position + 1)
    return dcg


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = dcg_at_k(ranked_ids, relevant_ids, k)
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(
        1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1)
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def grounding_precision(ranked_scores: list[float], threshold: float, k: int) -> float:
    """Fraction of the top-k candidates whose relevance score clears `threshold`.

    Distinct from precision@k: precision@k asks "are the top results actually
    relevant" (needs ground truth); this asks "how much of what would reach the
    LLM's context is even confidently relevant by the reranker's own calibrated
    score" — a leading proxy for hallucination risk that doesn't require labels,
    since it's exactly the signal `RerankingService`'s `min_relevance_score` gate
    acts on in production.
    """
    top_k_scores = ranked_scores[:k]
    if not top_k_scores:
        return 0.0
    return sum(1 for score in top_k_scores if score >= threshold) / len(top_k_scores)
