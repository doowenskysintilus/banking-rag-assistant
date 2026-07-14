"""Password hashing and JWT access/refresh token helpers.

Uses the `bcrypt` package directly rather than `passlib`: passlib's bcrypt backend
is unmaintained and misdetects modern bcrypt (>=4.1) releases, raising spurious
"password cannot be longer than 72 bytes" errors on ordinary, short passwords.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import bcrypt
from jose import JWTError, jwt

from backend.core.config import get_settings

settings = get_settings()

# bcrypt silently ignores bytes beyond the 72nd; truncate explicitly so behavior is
# deterministic and long (but not absurd) passwords don't raise on some backends.
_BCRYPT_MAX_BYTES = 72


def _prepare(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(plain_password: str) -> str:
    hashed = bcrypt.hashpw(_prepare(plain_password), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(plain_password), hashed_password.encode("utf-8"))
    except ValueError:
        return False


@dataclass(frozen=True)
class IssuedToken:
    """A freshly minted JWT plus the claims a caller needs to track it
    server-side (`backend.core.token_store`) without re-decoding it — `jti` to
    key a `RefreshSession`/`RevokedToken` row, `expires_at` for that row's
    expiry so it can be purged once the token itself would be rejected anyway.
    """

    token: str
    jti: str
    expires_at: datetime


def _create_token(
    subject: str,
    token_type: Literal["access", "refresh"],
    expires_delta: timedelta,
    extra_claims: dict[str, Any] | None = None,
) -> IssuedToken:
    now = datetime.now(timezone.utc)
    expires_at = now + expires_delta
    jti = uuid.uuid4().hex
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": expires_at,
        "jti": jti,
    }
    if extra_claims:
        payload.update(extra_claims)
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)
    return IssuedToken(token=token, jti=jti, expires_at=expires_at)


def create_access_token(
    subject: str, extra_claims: dict[str, Any] | None = None
) -> IssuedToken:
    return _create_token(
        subject,
        "access",
        timedelta(minutes=settings.access_token_expire_minutes),
        extra_claims,
    )


def create_refresh_token(subject: str) -> IssuedToken:
    return _create_token(
        subject, "refresh", timedelta(days=settings.refresh_token_expire_days)
    )


class TokenError(Exception):
    """Raised when a JWT is invalid, expired, or of the wrong type."""


def decode_token(
    token: str, expected_type: Literal["access", "refresh"] = "access"
) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as exc:
        raise TokenError("Invalid or expired token") from exc

    if payload.get("type") != expected_type:
        raise TokenError(f"Expected a {expected_type} token")
    return payload
