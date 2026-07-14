"""FastAPI application factory and entrypoint.

Run locally with:

    uvicorn backend.main:app --reload

The app is built via `create_app()` so tests can construct isolated instances with
overridden dependencies (`app.dependency_overrides`) instead of importing the
module-level singleton.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from backend.api.v1.router import api_router
from backend.core.config import get_settings
from backend.core.database import init_db
from backend.core.exceptions import register_exception_handlers
from backend.core.logging import (
    configure_logging,
    get_logger,
    new_request_id,
    request_id_ctx,
)
from backend.core.security_headers import SecurityHeadersMiddleware
from backend.monitoring.metrics import metrics_middleware, render_metrics

settings = get_settings()
configure_logging(level=settings.log_level, json_output=settings.log_json)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application_startup", extra={"environment": settings.app_env})
    if not settings.is_production:
        # In production, use Alembic migrations instead of create_all().
        await init_db()
    yield
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Production RAG assistant backend for internal banking knowledge queries.",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.is_production)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming_id = request.headers.get("x-request-id") or new_request_id()
        token = request_id_ctx.set(incoming_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["x-request-id"] = incoming_id
        return response

    app.middleware("http")(metrics_middleware)

    register_exception_handlers(app)

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return render_metrics()

    return app


app = create_app()
