"""End-to-end evaluation pipeline: build a dataset -> run providers -> persist
history -> write automatic reports. This is the "pipeline complet" entry point;
`backend.routers.evaluation` calls the same `run_and_persist` a triggered run
uses, and the dashboard queries the same `EvaluationRun` history this writes.

Runnable standalone against the static golden fixture set (no live database
documents, no running RAGEngine needed):
    python -m backend.evaluation.runner
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.evaluation.fixtures import GOLDEN_SAMPLES
from backend.evaluation.report import format_markdown_report, render_dashboard_html
from backend.evaluation.service import EvaluationService
from backend.evaluation.types import (
    EvalReport,
    EvalSample,
    MetricScore,
    SampleEvaluation,
)
from backend.models.evaluation import EvaluationRun, EvaluationSampleScore

REPORTS_DIR = Path("reports/evaluation")


async def load_report(db: AsyncSession, run_id: str) -> EvalReport | None:
    """Reconstructs an `EvalReport` from its persisted rows — used by
    `backend.api.v1.routers.evaluation` to serve a past run without needing to
    re-run any provider.
    """
    run = await db.get(EvaluationRun, run_id)
    if run is None:
        return None

    result = await db.execute(
        select(EvaluationSampleScore).where(EvaluationSampleScore.run_id == run_id)
    )
    sample_evaluations = [
        SampleEvaluation(
            sample=EvalSample(
                sample_id=row.sample_id,
                question=row.question,
                answer=row.answer,
                contexts=[],
                ground_truth=row.ground_truth,
            ),
            scores=[MetricScore(**s) for s in json.loads(row.scores_json)],
        )
        for row in result.scalars().all()
    ]

    return EvalReport(
        run_id=run.id,
        created_at=run.created_at,
        dataset_name=run.dataset_name,
        num_samples=run.num_samples,
        aggregate_scores=json.loads(run.aggregate_scores_json),
        providers_used=json.loads(run.providers_used_json),
        sample_evaluations=sample_evaluations,
        warnings=json.loads(run.warnings_json),
    )


async def fetch_run_history(
    db: AsyncSession,
    dataset_name: str,
    *,
    exclude_run_id: str | None = None,
    limit: int = 20,
) -> list[tuple[datetime, dict[str, float]]]:
    """Oldest-first list of `(created_at, aggregate_scores)` for prior runs of
    `dataset_name` — used to draw the dashboard's per-metric trend sparkline.
    """
    stmt = (
        select(EvaluationRun)
        .where(EvaluationRun.dataset_name == dataset_name)
        .order_by(EvaluationRun.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = [r for r in result.scalars().all() if r.id != exclude_run_id]
    rows.reverse()
    return [(r.created_at, json.loads(r.aggregate_scores_json)) for r in rows]


async def run_and_persist(
    db: AsyncSession,
    samples: list[EvalSample],
    dataset_name: str,
    *,
    service: EvaluationService | None = None,
    triggered_by_user_id: str | None = None,
    write_report_files: bool = True,
) -> EvalReport:
    service = service or EvaluationService()
    report = await service.evaluate(samples, dataset_name=dataset_name)

    db.add(
        EvaluationRun(
            id=report.run_id,
            triggered_by_user_id=triggered_by_user_id,
            dataset_name=report.dataset_name,
            num_samples=report.num_samples,
            aggregate_scores_json=json.dumps(report.aggregate_scores),
            providers_used_json=json.dumps(report.providers_used),
            warnings_json=json.dumps(report.warnings),
        )
    )
    for sample_eval in report.sample_evaluations:
        db.add(
            EvaluationSampleScore(
                run_id=report.run_id,
                sample_id=sample_eval.sample.sample_id,
                question=sample_eval.sample.question,
                answer=sample_eval.sample.answer,
                ground_truth=sample_eval.sample.ground_truth,
                scores_json=json.dumps(
                    [
                        {
                            "metric": s.metric,
                            "score": s.score,
                            "provider": s.provider,
                            "detail": s.detail,
                        }
                        for s in sample_eval.scores
                    ]
                ),
            )
        )
    await db.commit()

    if write_report_files:
        history = await fetch_run_history(
            db, dataset_name, exclude_run_id=report.run_id
        )
        _write_report_files(report, history)

    return report


def _write_report_files(
    report: EvalReport, history: list[tuple[datetime, dict[str, float]]]
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = REPORTS_DIR / f"{stamp}-{report.run_id}"
    base.with_suffix(".md").write_text(format_markdown_report(report), encoding="utf-8")
    base.with_suffix(".html").write_text(
        render_dashboard_html(report, history=history), encoding="utf-8"
    )


async def run_golden_fixture_benchmark(db: AsyncSession, **kwargs) -> EvalReport:
    """The offline/CI entry point — evaluates the static golden set in
    `fixtures.py`, requiring no live database documents or a running RAGEngine.
    """
    return await run_and_persist(
        db, GOLDEN_SAMPLES, dataset_name="golden-fixture-set", **kwargs
    )


async def _main() -> None:
    from backend.core.database import AsyncSessionLocal, init_db

    await init_db()
    async with AsyncSessionLocal() as db:
        report = await run_golden_fixture_benchmark(db)
    print(format_markdown_report(report))


if __name__ == "__main__":
    asyncio.run(_main())
