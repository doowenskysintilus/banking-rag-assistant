"""The final ranking pass: adjusts relevance scores with business-aware signals
that neither dense similarity nor BM25 nor the cross-encoder know about — a more
recently uploaded policy is more likely to be current guidance, and a chunk that
carries its structural context (an Article/Clause breadcrumb) is more likely to be
precise and unambiguous than a bare paragraph. Pure Python, no model, cheap.

Each rule is independent and additive, so the effect of any one of them is easy to
reason about (and to A/B by tweaking a single weight in Settings).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_core.documents import Document

_SCORE_METADATA_KEYS = ("rerank_score", "score")


@dataclass(frozen=True)
class BoostRule:
    name: str
    weight: float
    # metadata -> raw signal, expected roughly in [0, 1] (weight scales it from there)
    signal: Callable[[dict], float]


def _recency_signal(half_life_days: float) -> Callable[[dict], float]:
    def _signal(metadata: dict) -> float:
        created_at = metadata.get("created_at")
        if not created_at:
            return 0.0
        try:
            created = datetime.fromisoformat(created_at)
        except (TypeError, ValueError):
            return 0.0
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
        if age_days < 0 or half_life_days <= 0:
            return 1.0
        return 0.5 ** (age_days / half_life_days)

    return _signal


def _structural_context_signal(metadata: dict) -> float:
    return 1.0 if metadata.get("header_path") else 0.0


def _base_score(metadata: dict) -> float:
    for key in _SCORE_METADATA_KEYS:
        if key in metadata:
            return float(metadata[key])
    return 0.0


class IntelligentRanker:
    def __init__(self, rules: list[BoostRule] | None = None) -> None:
        self._rules = rules if rules is not None else []

    @classmethod
    def with_default_rules(
        cls,
        *,
        recency_weight: float = 0.05,
        recency_half_life_days: float = 180.0,
        structural_context_weight: float = 0.02,
    ) -> "IntelligentRanker":
        rules = []
        if recency_weight:
            rules.append(
                BoostRule(
                    name="recency",
                    weight=recency_weight,
                    signal=_recency_signal(recency_half_life_days),
                )
            )
        if structural_context_weight:
            rules.append(
                BoostRule(
                    name="structural_context",
                    weight=structural_context_weight,
                    signal=_structural_context_signal,
                )
            )
        return cls(rules)

    def rank(self, documents: list[Document]) -> list[Document]:
        """Returns a new, re-sorted list — never mutates the input documents."""
        scored: list[tuple[Document, float, dict[str, float]]] = []
        for doc in documents:
            base = _base_score(doc.metadata)
            boosts = {
                rule.name: rule.weight * rule.signal(doc.metadata)
                for rule in self._rules
            }
            final_score = base + sum(boosts.values())
            scored.append((doc, final_score, boosts))

        scored.sort(key=lambda item: item[1], reverse=True)

        return [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "final_score": final_score, "boosts": boosts},
            )
            for doc, final_score, boosts in scored
        ]
