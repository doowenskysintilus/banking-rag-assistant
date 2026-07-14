"""Confidence scoring for a generated answer.

Deliberately *retrieval-grounded* rather than LLM-self-reported: asking a model to
rate its own confidence is known to be unreliable (models are frequently
overconfident, especially while hallucinating). Instead this scores how strong the
evidence actually was — the same calibrated relevance scores
`backend.retrieval.reranking` already uses to decide whether to drop a chunk
entirely (see `enterprise_reranker_min_relevance_score`) — and turns that into a
single number plus a coarse label a UI can render as a badge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ConfidenceLevel = Literal["high", "medium", "low"]

_HIGH_THRESHOLD = 0.75
_MEDIUM_THRESHOLD = 0.4


@dataclass(frozen=True)
class ConfidenceAssessment:
    score: float
    level: ConfidenceLevel


def assess_confidence(
    relevance_scores: list[float], low_confidence_signal: bool = False
) -> ConfidenceAssessment:
    """`relevance_scores` are the final (post-rerank, post-filter) source scores
    actually used to build the answer's context. `low_confidence_signal` lets a
    caller force "low" even with scores present (e.g. the reranker's own
    min-relevance-score gate reported nothing cleared its bar).
    """
    if low_confidence_signal or not relevance_scores:
        return ConfidenceAssessment(score=0.0, level="low")

    ranked = sorted(relevance_scores, reverse=True)
    top_score = ranked[0]
    top_three = ranked[:3]
    avg_top_three = sum(top_three) / len(top_three)

    # Weighted toward the single best source (what ultimately grounds most of the
    # answer) while still rewarding having multiple corroborating sources.
    combined = round(0.6 * top_score + 0.4 * avg_top_three, 4)
    combined = min(1.0, max(0.0, combined))

    if combined >= _HIGH_THRESHOLD:
        level: ConfidenceLevel = "high"
    elif combined >= _MEDIUM_THRESHOLD:
        level = "medium"
    else:
        level = "low"

    return ConfidenceAssessment(score=combined, level=level)
