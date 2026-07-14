"""`EvaluationService` — the entry point the rest of the app depends on.

Runs every configured provider (RAGAS, DeepEval) only for the metrics
`backend.evaluation.selector` actually assigned to it, then merges each
provider's per-sample scores into one `EvalReport`. A metric with no available
provider is skipped and recorded in `EvalReport.warnings` rather than silently
omitted — mirrors how `backend.retrieval.reranking.service` surfaces
low-confidence retrieval instead of hiding it.
"""

from __future__ import annotations

import statistics
import uuid
from datetime import datetime, timezone

from backend.core.config import Settings, get_settings
from backend.evaluation.providers.base import EvaluationProvider
from backend.evaluation.providers.deepeval_provider import DeepEvalEvaluationProvider
from backend.evaluation.providers.ragas_provider import RagasEvaluationProvider
from backend.evaluation.selector import probe_availability, select_metric_providers
from backend.evaluation.types import (
    EvalReport,
    EvalSample,
    MetricScore,
    SampleEvaluation,
)


def build_default_providers(settings: Settings) -> list[EvaluationProvider]:
    return [RagasEvaluationProvider(settings), DeepEvalEvaluationProvider(settings)]


class EvaluationService:
    def __init__(
        self,
        providers: list[EvaluationProvider] | None = None,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._providers = (
            providers if providers is not None else build_default_providers(settings)
        )

    async def evaluate(
        self, samples: list[EvalSample], dataset_name: str = "ad-hoc"
    ) -> EvalReport:
        run_id = str(uuid.uuid4())
        warnings: list[str] = []

        if not samples:
            return EvalReport(
                run_id=run_id,
                created_at=datetime.now(timezone.utc),
                dataset_name=dataset_name,
                num_samples=0,
                aggregate_scores={},
                providers_used={},
                sample_evaluations=[],
                warnings=["No samples were provided — nothing was evaluated."],
            )

        availability = probe_availability(self._providers)
        assignment = select_metric_providers(self._providers, availability)

        for provider_name, available in availability.items():
            if not available:
                warnings.append(
                    f"Provider '{provider_name}' is unavailable in this environment — its metrics were skipped where no fallback existed."
                )

        by_name = {p.name: p for p in self._providers}
        metrics_per_provider: dict[str, list[str]] = {}
        for metric, provider_name in assignment.items():
            metrics_per_provider.setdefault(provider_name, []).append(metric)

        # per_sample_scores[metric][sample_index] = float | None
        per_sample_scores: dict[str, list[float | None]] = {}
        for provider_name, metric_names in metrics_per_provider.items():
            provider = by_name[provider_name]
            result = await provider.evaluate(samples, metric_names)
            per_sample_scores.update(result)

        for metric in set(assignment) - set(per_sample_scores):
            # Defensive: a provider claimed a metric via select_metric_providers
            # but didn't return it — treat as entirely missing rather than crash.
            per_sample_scores[metric] = [None] * len(samples)

        for metric in (
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevancy",
            "hallucination_rate",
        ):
            if metric not in assignment:
                warnings.append(
                    f"No available provider supports '{metric}' — it was skipped for this run."
                )

        sample_evaluations: list[SampleEvaluation] = []
        for i, sample in enumerate(samples):
            scores = [
                MetricScore(metric=metric, score=values[i], provider=assignment[metric])
                for metric, values in per_sample_scores.items()
                if values[i] is not None
            ]
            sample_evaluations.append(SampleEvaluation(sample=sample, scores=scores))

        aggregate_scores: dict[str, float] = {}
        for metric, values in per_sample_scores.items():
            clean = [v for v in values if v is not None]
            if clean:
                aggregate_scores[metric] = round(statistics.mean(clean), 4)
            else:
                warnings.append(
                    f"Every sample failed scoring for '{metric}' — no aggregate could be computed."
                )

        return EvalReport(
            run_id=run_id,
            created_at=datetime.now(timezone.utc),
            dataset_name=dataset_name,
            num_samples=len(samples),
            aggregate_scores=aggregate_scores,
            providers_used=assignment,
            sample_evaluations=sample_evaluations,
            warnings=warnings,
        )


_service_singleton: EvaluationService | None = None


def get_evaluation_service() -> EvaluationService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = EvaluationService()
    return _service_singleton
