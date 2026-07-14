"""Read-side access to the compliance audit trail — writes all go through
`backend.core.audit.append_audit_log` from wherever the event happens
(auth, user management, RAG queries); this is just listing/verification for
the admin-facing endpoints in `backend.api.v1.routers.audit`.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.audit import verify_audit_chain
from backend.models.audit_log import AuditLog


class AuditService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def list_entries(
        self,
        *,
        user_id: str | None = None,
        action: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditLog], int]:
        stmt = select(AuditLog)
        count_stmt = select(func.count()).select_from(AuditLog)
        if user_id is not None:
            stmt = stmt.where(AuditLog.user_id == user_id)
            count_stmt = count_stmt.where(AuditLog.user_id == user_id)
        if action is not None:
            stmt = stmt.where(AuditLog.action == action)
            count_stmt = count_stmt.where(AuditLog.action == action)

        total = (await self._db.execute(count_stmt)).scalar_one()
        result = await self._db.execute(
            stmt.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all()), total

    async def verify_chain(self) -> list[str]:
        return await verify_audit_chain(self._db)
