"""Read access to the compliance audit trail — listing and tamper-evidence
verification (see `backend.core.audit`). `Permission.AUDIT_READ` is
ADMIN-only per `backend.core.permissions.ROLE_PERMISSIONS`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.core.dependencies import get_audit_service, require_permissions
from backend.core.permissions import Permission
from backend.models.user import User
from backend.schemas.audit import AuditChainVerification, AuditLogEntry, AuditLogList
from backend.services.audit_service import AuditService

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/logs", response_model=AuditLogList, summary="List audit trail entries")
async def list_logs(
    user_id: str | None = Query(default=None),
    action: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _current_user: User = require_permissions(Permission.AUDIT_READ),
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditLogList:
    items, total = await audit_service.list_entries(
        user_id=user_id, action=action, limit=limit, offset=offset
    )
    return AuditLogList(
        items=[AuditLogEntry.model_validate(e) for e in items], total=total
    )


@router.get(
    "/verify",
    response_model=AuditChainVerification,
    summary="Verify the audit trail's tamper-evidence chain",
)
async def verify(
    _current_user: User = require_permissions(Permission.AUDIT_READ),
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditChainVerification:
    problems = await audit_service.verify_chain()
    items, total = await audit_service.list_entries(limit=1)
    return AuditChainVerification(
        intact=not problems, checked_entries=total, problems=problems
    )
