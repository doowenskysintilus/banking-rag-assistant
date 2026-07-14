"""Baseline response hardening headers — cheap, standard, and expected by any
banking security review (OWASP Secure Headers Project). This doesn't replace
TLS termination (a deployment/infra concern, not something this process can
enforce on itself) but does tell browsers to insist on it via HSTS once it's
in place, and closes off a few classic content-sniffing/framing/referrer-leak
issues regardless.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, hsts: bool = True) -> None:  # noqa: ANN001 - Starlette ASGI app
        super().__init__(app)
        self._hsts = hsts

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        # This is an API, not a page renderer — a strict CSP that disallows
        # everything is correct here; the interactive docs (/docs, /redoc) are
        # disabled in production (see backend.main.create_app) specifically
        # because they'd need a looser policy to function.
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )

        if self._hsts:
            # `preload` is deliberately omitted — that's a one-way door (submission
            # to browsers' hardcoded preload list) that should be an explicit,
            # separate decision once the domain's TLS setup is final, not a default.
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains"
            )

        return response
