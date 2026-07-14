"""Shared types for the RAG evaluation framework.

`EvalSample` is the unit fed to every metric provider: a question, the RAG
pipeline's actual answer, the context chunks it actually retrieved, and a
reference (ground truth) answer to compare against. `MetricScore.score` is
always normalized to [0, 1] with "higher is better" — including
`hallucination_rate`, which providers must return as `1 - <raw hallucination
fraction>` (i.e. a "faithfulness-to-non-hallucination" score) so the dashboard
can apply one "higher is better" rule uniformly rather than special-casing one
metric. The raw fraction is preserved in `MetricScore.detail` for transparency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "hallucination_rate",
)


@dataclass(frozen=True)
class EvalSample:
    """One question run through the RAG pipeline, plus the reference answer it
    is judged against. `contexts` are the actual retrieved/reranked chunks the
    LLM saw — not the ideal ones — since faithfulness/context metrics must
    judge what really happened, not an idealized retrieval.
    """

    sample_id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


@dataclass(frozen=True)
class MetricScore:
    metric: str
    score: float
    provider: str
    detail: str = ""


@dataclass(frozen=True)
class SampleEvaluation:
    sample: EvalSample
    scores: list[MetricScore]

    def score_for(self, metric: str) -> float | None:
        for s in self.scores:
            if s.metric == metric:
                return s.score
        return None


@dataclass(frozen=True)
class EvalReport:
    run_id: str
    created_at: datetime
    dataset_name: str
    num_samples: int
    aggregate_scores: dict[str, float]
    providers_used: dict[str, str]
    sample_evaluations: list[SampleEvaluation]
    warnings: list[str] = field(default_factory=list)
