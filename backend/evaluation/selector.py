"""Decides which provider computes each of the five required metrics.

Unlike `backend.retrieval.reranking.selector` (picks *one* provider for
everything), this composes RAGAS and DeepEval side by side — each covers a
different subset of the five metrics (see the two provider modules for why).
`select_metric_providers` still follows the same "auto" priority-list pattern:
if a metric's preferred provider isn't available, fall back to any other
provider that also supports it; if none do, the metric is skipped (never
silently scored as 0 — a missing evaluation must read as missing, not as a
failing grade).
"""

from __future__ import annotations

from backend.evaluation.providers.base import EvaluationProvider
from backend.evaluation.types import METRIC_NAMES

# Preferred provider per metric — see backend.evaluation.providers.ragas_provider
# and .deepeval_provider for why the five required metrics are split this way.
_METRIC_PROVIDER_PRIORITY: dict[str, tuple[str, ...]] = {
    "faithfulness": ("ragas", "deepeval"),
    "context_precision": ("ragas", "deepeval"),
    "context_recall": ("ragas", "deepeval"),
    "answer_relevancy": ("deepeval", "ragas"),
    "hallucination_rate": ("deepeval",),
}


def probe_availability(providers: list[EvaluationProvider]) -> dict[str, bool]:
    return {provider.name: provider.is_available for provider in providers}


def select_metric_providers(
    providers: list[EvaluationProvider], availability: dict[str, bool] | None = None
) -> dict[str, str]:
    """Returns `{metric_name: provider_name}` for every metric that at least one
    available provider actually supports. A metric absent from the result means
    no available provider covers it — callers must surface that as a warning,
    not silently drop the metric from the report.
    """
    availability = (
        availability if availability is not None else probe_availability(providers)
    )
    by_name = {p.name: p for p in providers}

    assignment: dict[str, str] = {}
    for metric in METRIC_NAMES:
        priority = _METRIC_PROVIDER_PRIORITY.get(metric, ())
        for provider_name in priority:
            provider = by_name.get(provider_name)
            if provider is None or metric not in provider.supported_metrics:
                continue
            if availability.get(provider_name):
                assignment[metric] = provider_name
                break
    return assignment
