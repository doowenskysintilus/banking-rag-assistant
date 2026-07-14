"""Application-wide exception hierarchy and FastAPI exception handlers.

All API errors follow a single JSON envelope:

    {
        "error": {
            "code": "resource_not_found",
            "message": "Document abc123 was not found",
            "request_id": "a1b2c3d4e5f6"
        }
    }
"""

from __future__ import annotations

from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for all application-raised errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "not_found"


class ValidationAppError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    error_code = "validation_error"


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "unauthorized"


class ForbiddenError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    error_code = "forbidden"


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    error_code = "conflict"


class RateLimitedError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "rate_limited"


class ExternalServiceError(AppError):
    """Raised when a downstream dependency (OpenAI API, vector store, DB) fails."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "external_service_error"


def _error_response(
    status_code: int, error_code: str, message: str, details: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "request_id": request_id_ctx.get(),
                **({"details": details} if details else {}),
            }
        },
    )


def register_exception_handlers(app) -> None:  # noqa: ANN001 - FastAPI app
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "app_error",
            extra={"error_code": exc.error_code, "path": request.url.path},
        )
        return _error_response(
            exc.status_code, exc.error_code, exc.message, exc.details
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info(
            "request_validation_error",
            extra={"path": request.url.path, "errors": exc.errors()},
        )
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "validation_error",
            "Request validation failed",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("unhandled_exception", extra={"path": request.url.path})
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "An unexpected error occurred. Please try again later.",
        )
