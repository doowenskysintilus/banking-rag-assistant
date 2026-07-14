"""Liveness / readiness probes for orchestrators (Kubernetes, load balancers)."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import get_settings
from backend.core.database import get_db
from backend.schemas.common import HealthStatus, ReadinessCheck, ReadinessStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus, summary="Liveness probe")
async def health() -> HealthStatus:
    settings = get_settings()
    return HealthStatus(status="ok", version="1.0.0", environment=settings.app_env)


@router.get("/health/ready", response_model=ReadinessStatus, summary="Readiness probe")
async def readiness(db: AsyncSession = Depends(get_db)) -> ReadinessStatus:
    checks: list[ReadinessCheck] = []

    try:
        await db.execute(text("SELECT 1"))
        checks.append(ReadinessCheck(name="database", healthy=True))
    except Exception as exc:  # noqa: BLE001 - readiness must never raise
        checks.append(ReadinessCheck(name="database", healthy=False, detail=str(exc)))

    overall = "ok" if all(c.healthy for c in checks) else "degraded"
    return ReadinessStatus(status=overall, checks=checks)
