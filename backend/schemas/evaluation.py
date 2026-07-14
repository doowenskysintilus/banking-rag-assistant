from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, Field

from backend.models.evaluation import EvaluationRun


class LabeledQuestionIn(BaseModel):
    """One hand-labeled question for a *live* evaluation run — answered by the
    caller's own RAGEngine against their own documents (see
    backend.evaluation.dataset.EvalDatasetBuilder). Omit `questions` entirely
    on `EvalRunRequest` to evaluate the static, no-database golden fixture set
    instead.
    """

    sample_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=2000)
    ground_truth: str = Field(min_length=1, max_length=4000)


class EvalRunRequest(BaseModel):
    dataset_name: str = Field(default="golden-fixture-set", max_length=100)
    questions: list[LabeledQuestionIn] | None = Field(
        default=None,
        description="If omitted, evaluates the static golden fixture set. If provided, runs each "
        "question through the live RAG engine against the caller's own documents first.",
    )


class MetricScorePublic(BaseModel):
    metric: str
    score: float
    provider: str


class SampleEvaluationPublic(BaseModel):
    sample_id: str
    question: str
    answer: str
    ground_truth: str
    scores: list[MetricScorePublic]


class EvalReportPublic(BaseModel):
    run_id: str
    created_at: datetime
    dataset_name: str
    num_samples: int
    aggregate_scores: dict[str, float]
    providers_used: dict[str, str]
    warnings: list[str]
    sample_evaluations: list[SampleEvaluationPublic] = Field(default_factory=list)


class EvalRunSummary(BaseModel):
    id: str
    dataset_name: str
    num_samples: int
    aggregate_scores: dict[str, float]
    warnings: list[str]
    created_at: datetime

    @classmethod
    def from_row(cls, run: EvaluationRun) -> EvalRunSummary:
        return cls(
            id=run.id,
            dataset_name=run.dataset_name,
            num_samples=run.num_samples,
            aggregate_scores=json.loads(run.aggregate_scores_json),
            warnings=json.loads(run.warnings_json),
            created_at=run.created_at,
        )
