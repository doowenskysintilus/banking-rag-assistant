"""Server-side state for otherwise-stateless JWTs — required to support
immediate revocation (logout, compromised-token response) and refresh-token
rotation with reuse detection, neither of which a bare signed JWT can do on
its own.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base
from backend.core.datetime_utils import ensure_utc


class RefreshSession(Base):
    """One row per issued refresh token — the source of truth for whether a
    refresh token is still valid. `id` is the token's own `jti`, so validating
    a refresh token is a single primary-key lookup.

    Rotation: each successful `/auth/token` refresh grant revokes the
    presented session (`revoked_at` set, `replaced_by_jti` pointing at the new
    one) and creates a new row. If a token whose session is already revoked is
    presented again, that's a reuse of a stolen/replayed refresh token — see
    `AuthService.refresh`, which responds by revoking every session for that
    user, not just the one presented.
    """

    __tablename__ = "refresh_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    replaced_by_jti: Mapped[str | None] = mapped_column(String(36), nullable=True)

    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None and ensure_utc(self.expires_at) > datetime.now(
            timezone.utc
        )


class RevokedToken(Base):
    """Denylist for immediate access-token invalidation (logout, or a refresh
    session being force-revoked) — a bare JWT can't be "un-issued", so a
    signature- and expiry-valid token whose `jti` shows up here is rejected
    anyway (see `backend.core.dependencies.get_current_user`). Rows are safe to
    purge once `expires_at` has passed: signature verification already rejects
    an expired token regardless of this table (see `purge_expired_revocations`
    in `backend.core.security`).
    """

    __tablename__ = "revoked_tokens"

    jti: Mapped[str] = mapped_column(String(36), primary_key=True)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    revoked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    reason: Mapped[str] = mapped_column(String(100), nullable=False, default="logout")
