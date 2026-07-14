"""DB-backed counterpart to `backend.core.security` (which stays pure and
DB-free: JWT encode/decode, password hashing). This is where the *stateful*
half of "enterprise-grade JWT" lives — refresh-token session tracking (for
rotation + reuse detection) and an access-token revocation denylist (for
immediate logout) — both backed by `backend.models.session`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.session import RefreshSession, RevokedToken


async def create_refresh_session(
    db: AsyncSession,
    *,
    jti: str,
    user_id: str,
    expires_at: datetime,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> RefreshSession:
    session = RefreshSession(
        id=jti,
        user_id=user_id,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(session)
    return session


async def get_refresh_session(db: AsyncSession, jti: str) -> RefreshSession | None:
    return await db.get(RefreshSession, jti)


async def revoke_refresh_session(
    db: AsyncSession, jti: str, *, replaced_by_jti: str | None = None
) -> None:
    session = await db.get(RefreshSession, jti)
    if session is not None and session.revoked_at is None:
        session.revoked_at = datetime.now(timezone.utc)
        session.replaced_by_jti = replaced_by_jti


async def revoke_all_refresh_sessions_for_user(db: AsyncSession, user_id: str) -> int:
    """Used on password change and on refresh-token reuse detection ("logout
    everywhere"). Returns how many sessions were actually revoked.
    """
    result = await db.execute(
        select(RefreshSession).where(
            RefreshSession.user_id == user_id, RefreshSession.revoked_at.is_(None)
        )
    )
    sessions = result.scalars().all()
    now = datetime.now(timezone.utc)
    for session in sessions:
        session.revoked_at = now
    return len(sessions)


async def revoke_access_token(
    db: AsyncSession, *, jti: str, expires_at: datetime, reason: str = "logout"
) -> None:
    db.add(RevokedToken(jti=jti, expires_at=expires_at, reason=reason))


async def is_token_revoked(db: AsyncSession, jti: str) -> bool:
    return await db.get(RevokedToken, jti) is not None


async def purge_expired_revocations(db: AsyncSession) -> int:
    """Deletes `RevokedToken` rows past their own `expires_at` — safe because a
    token past its JWT `exp` claim is already rejected by signature
    verification regardless of whether it's still listed here. Call this
    periodically (e.g. from a scheduled maintenance task); it isn't required
    for correctness, only to bound the table's growth.
    """
    result = await db.execute(
        delete(RevokedToken).where(RevokedToken.expires_at < datetime.now(timezone.utc))
    )
    return result.rowcount or 0
