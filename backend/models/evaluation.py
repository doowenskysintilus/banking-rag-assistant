"""Persisted history for the RAG evaluation framework (backend.evaluation) — one
row per run plus one row per evaluated sample, so the dashboard can show trends
across runs rather than only the latest result.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    triggered_by_user_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    dataset_name: Mapped[str] = mapped_column(String(100), nullable=False)
    num_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # JSON-encoded {"faithfulness": 0.91, "context_precision": 0.83, ...} — the mean
    # score per metric across every sample in this run. See backend.evaluation.types.
    aggregate_scores_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )
    # JSON-encoded {"faithfulness": "ragas", "hallucination_rate": "deepeval", ...} —
    # which provider actually computed each metric (may fall back or be skipped —
    # see backend.evaluation.selector).
    providers_used_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # JSON-encoded list of strings — e.g. a metric skipped because no provider for
    # it was available in this environment.
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EvaluationSampleScore(Base):
    __tablename__ = "evaluation_sample_scores"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    sample_id: Mapped[str] = mapped_column(String(200), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # JSON-encoded list of {"metric", "score", "provider", "detail"} — see
    # backend.evaluation.types.MetricScore.
    scores_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
