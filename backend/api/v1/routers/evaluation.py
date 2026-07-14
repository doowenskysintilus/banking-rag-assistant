"""RAG evaluation framework endpoints — trigger a scored run, browse run
history, and view the self-contained HTML dashboard. Admin-only: this exposes
retrieval-quality internals (and, for live runs, actual answer text) that
aren't relevant to normal chat users.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.dependencies import (
    get_db,
    get_evaluation_service_dep,
    get_rag_engine,
    require_permissions,
)
from backend.core.exceptions import NotFoundError
from backend.core.permissions import Permission
from backend.evaluation.dataset import EvalDatasetBuilder, LabeledQuestion
from backend.evaluation.fixtures import GOLDEN_SAMPLES
from backend.evaluation.report import render_dashboard_html
from backend.evaluation.runner import fetch_run_history, load_report, run_and_persist
from backend.evaluation.service import EvaluationService
from backend.models.evaluation import EvaluationRun
from backend.models.user import User
from backend.rag.engine import RAGEngine
from backend.schemas.evaluation import (
    EvalReportPublic,
    EvalRunRequest,
    EvalRunSummary,
    MetricScorePublic,
    SampleEvaluationPublic,
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _to_public(report) -> EvalReportPublic:
    return EvalReportPublic(
        run_id=report.run_id,
        created_at=report.created_at,
        dataset_name=report.dataset_name,
        num_samples=report.num_samples,
        aggregate_scores=report.aggregate_scores,
        providers_used=report.providers_used,
        warnings=report.warnings,
        sample_evaluations=[
            SampleEvaluationPublic(
                sample_id=s.sample.sample_id,
                question=s.sample.question,
                answer=s.sample.answer,
                ground_truth=s.sample.ground_truth,
                scores=[
                    MetricScorePublic(
                        metric=m.metric, score=m.score, provider=m.provider
                    )
                    for m in s.scores
                ],
            )
            for s in report.sample_evaluations
        ],
    )


@router.post(
    "/runs",
    response_model=EvalReportPublic,
    summary="Trigger an evaluation run (golden fixture set, or live against the caller's own documents)",
)
async def trigger_run(
    payload: EvalRunRequest,
    current_user: User = require_permissions(Permission.EVALUATION_RUN),
    db: AsyncSession = Depends(get_db),
    rag_engine: RAGEngine = Depends(get_rag_engine),
    service: EvaluationService = Depends(get_evaluation_service_dep),
) -> EvalReportPublic:
    if payload.questions:
        builder = EvalDatasetBuilder(rag_engine)
        labeled = [
            LabeledQuestion(
                sample_id=q.sample_id, question=q.question, ground_truth=q.ground_truth
            )
            for q in payload.questions
        ]
        samples = await builder.build(db, current_user.id, labeled)
    else:
        samples = GOLDEN_SAMPLES

    report = await run_and_persist(
        db,
        samples,
        payload.dataset_name,
        service=service,
        triggered_by_user_id=current_user.id,
    )
    return _to_public(report)


@router.get(
    "/runs", response_model=list[EvalRunSummary], summary="List past evaluation runs"
)
async def list_runs(
    dataset_name: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    current_user: User = require_permissions(Permission.EVALUATION_READ),
    db: AsyncSession = Depends(get_db),
) -> list[EvalRunSummary]:
    stmt = select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
    if dataset_name:
        stmt = stmt.where(EvaluationRun.dataset_name == dataset_name)
    result = await db.execute(stmt)
    return [EvalRunSummary.from_row(r) for r in result.scalars().all()]


@router.get(
    "/runs/{run_id}",
    response_model=EvalReportPublic,
    summary="Fetch one evaluation run's full report",
)
async def get_run(
    run_id: str,
    current_user: User = require_permissions(Permission.EVALUATION_READ),
    db: AsyncSession = Depends(get_db),
) -> EvalReportPublic:
    report = await load_report(db, run_id)
    if report is None:
        raise NotFoundError(f"Evaluation run {run_id} was not found")
    return _to_public(report)


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Self-contained HTML dashboard for the latest run",
)
async def dashboard(
    dataset_name: str = Query(default="golden-fixture-set"),
    current_user: User = require_permissions(Permission.EVALUATION_READ),
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    stmt = (
        select(EvaluationRun)
        .where(EvaluationRun.dataset_name == dataset_name)
        .order_by(EvaluationRun.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    latest = result.scalar_one_or_none()
    if latest is None:
        return HTMLResponse(
            f"<p>No evaluation runs yet for dataset '{dataset_name}'. "
            f"POST /api/v1/evaluation/runs to create one.</p>",
            status_code=404,
        )

    report = await load_report(db, latest.id)
    history = await fetch_run_history(db, dataset_name, exclude_run_id=latest.id)
    return HTMLResponse(render_dashboard_html(report, history=history))
