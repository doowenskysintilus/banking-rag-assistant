from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from backend.core.security_headers import SecurityHeadersMiddleware


def _make_app(*, hsts: bool) -> Starlette:
    async def homepage(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", homepage)])
    app.add_middleware(SecurityHeadersMiddleware, hsts=hsts)
    return app


def test_adds_baseline_security_headers():
    client = TestClient(_make_app(hsts=True))
    response = client.get("/")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "geolocation=()" in response.headers["Permissions-Policy"]
    assert (
        response.headers["Content-Security-Policy"]
        == "default-src 'none'; frame-ancestors 'none'"
    )


def test_hsts_header_present_when_enabled():
    client = TestClient(_make_app(hsts=True))
    response = client.get("/")
    assert "Strict-Transport-Security" in response.headers
    assert "max-age=63072000" in response.headers["Strict-Transport-Security"]


def test_hsts_header_absent_when_disabled():
    client = TestClient(_make_app(hsts=False))
    response = client.get("/")
    assert "Strict-Transport-Security" not in response.headers
