"""Admin user management: listing, role changes, activation/deactivation, and
manual unlock — kept separate from `AuthService` (authentication concerns:
login, tokens, password) the same way `DocumentService`/`ChatService` are
split by concern rather than growing one god-service.

Every mutation here is security-sensitive by definition (it changes who can
do what) and is audited via `backend.core.audit.append_audit_log`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.audit import append_audit_log
from backend.core.datetime_utils import ensure_utc
from backend.core.exceptions import ForbiddenError, NotFoundError
from backend.core.logging import get_logger
from backend.models.user import User, UserRole

logger = get_logger(__name__)


class UserService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def list_users(
        self,
        *,
        role: UserRole | None = None,
        is_active: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[User], int]:
        stmt = select(User)
        count_stmt = select(func.count()).select_from(User)
        if role is not None:
            stmt = stmt.where(User.role == role)
            count_stmt = count_stmt.where(User.role == role)
        if is_active is not None:
            stmt = stmt.where(User.is_active == is_active)
            count_stmt = count_stmt.where(User.is_active == is_active)

        total = (await self._db.execute(count_stmt)).scalar_one()
        result = await self._db.execute(
            stmt.order_by(User.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all()), total

    async def get_user(self, user_id: str) -> User:
        user = await self._db.get(User, user_id)
        if user is None:
            raise NotFoundError(f"User {user_id} was not found")
        return user

    async def update_role(
        self,
        *,
        actor: User,
        target_user_id: str,
        new_role: UserRole,
        ip_address: str | None = None,
    ) -> User:
        if actor.id == target_user_id:
            raise ForbiddenError(
                "You cannot change your own role — ask another administrator to do it"
            )

        user = await self.get_user(target_user_id)
        previous_role = user.role
        user.role = new_role
        await self._db.flush()

        await append_audit_log(
            self._db,
            user_id=actor.id,
            action="user_role_changed",
            resource=target_user_id,
            detail=f"role changed from {previous_role.value} to {new_role.value}",
            ip_address=ip_address,
        )
        await self._db.commit()
        await self._db.refresh(user)
        logger.info(
            "user_role_changed",
            extra={
                "actor_id": actor.id,
                "target_user_id": target_user_id,
                "new_role": new_role.value,
            },
        )
        return user

    async def set_active(
        self,
        *,
        actor: User,
        target_user_id: str,
        is_active: bool,
        reason: str | None = None,
        ip_address: str | None = None,
    ) -> User:
        if actor.id == target_user_id:
            raise ForbiddenError("You cannot activate or deactivate your own account")

        user = await self.get_user(target_user_id)
        user.is_active = is_active
        await self._db.flush()

        await append_audit_log(
            self._db,
            user_id=actor.id,
            action="user_activated" if is_active else "user_deactivated",
            resource=target_user_id,
            detail=reason,
            ip_address=ip_address,
        )
        await self._db.commit()
        await self._db.refresh(user)
        logger.info(
            "user_active_state_changed",
            extra={
                "actor_id": actor.id,
                "target_user_id": target_user_id,
                "is_active": is_active,
            },
        )
        return user

    async def unlock_user(
        self,
        *,
        actor: User,
        target_user_id: str,
        reason: str | None = None,
        ip_address: str | None = None,
    ) -> User:
        """Clears a brute-force lockout early (see `AuthService.authenticate`) —
        for a legitimate user who got locked out and verified their identity
        through another channel.
        """
        user = await self.get_user(target_user_id)
        user.failed_login_attempts = 0
        user.locked_until = None
        await self._db.flush()

        await append_audit_log(
            self._db,
            user_id=actor.id,
            action="user_unlocked",
            resource=target_user_id,
            detail=reason,
            ip_address=ip_address,
        )
        await self._db.commit()
        await self._db.refresh(user)
        logger.info(
            "user_unlocked",
            extra={"actor_id": actor.id, "target_user_id": target_user_id},
        )
        return user

    @staticmethod
    def is_locked(user: User) -> bool:
        return user.locked_until is not None and ensure_utc(
            user.locked_until
        ) > datetime.now(timezone.utc)
