"""Prometheus metrics for HTTP traffic, RAG pipeline latency, and LLM token usage."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
)

RAG_QUERY_DURATION_SECONDS = Histogram(
    "rag_query_duration_seconds",
    "End-to-end RAG query latency (retrieval + generation)",
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Tokens consumed calling the LLM",
    ["direction"],  # "input" | "output"
)

DOCUMENT_INGESTION_TOTAL = Counter(
    "document_ingestion_total",
    "Documents processed by the ingestion pipeline",
    ["status"],  # "success" | "failure"
)


async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    route = request.scope.get("route")
    path_template = getattr(route, "path", request.url.path)

    HTTP_REQUESTS_TOTAL.labels(
        request.method, path_template, response.status_code
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(request.method, path_template).observe(
        duration
    )
    return response


def render_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
