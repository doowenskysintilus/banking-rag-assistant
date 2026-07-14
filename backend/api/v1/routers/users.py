"""Admin user management: list, view, change role, activate/deactivate, unlock.

Every mutating endpoint here requires `Permission.USER_MANAGE` (ADMIN-only per
`backend.core.permissions.ROLE_PERMISSIONS`) and is written to the audit trail
by `UserService` — see that module for the actual business rules (e.g. an
admin cannot change their own role or deactivate their own account).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from backend.core.dependencies import get_user_service, require_permissions
from backend.core.permissions import Permission
from backend.models.user import User, UserRole
from backend.schemas.user import (
    SetUserActiveRequest,
    UnlockUserRequest,
    UpdateUserRoleRequest,
    UserAdminView,
    UserList,
)
from backend.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["users"])


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


@router.get("", response_model=UserList, summary="List users")
async def list_users(
    role: UserRole | None = Query(default=None),
    is_active: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _current_user: User = require_permissions(Permission.USER_READ),
    user_service: UserService = Depends(get_user_service),
) -> UserList:
    items, total = await user_service.list_users(
        role=role, is_active=is_active, limit=limit, offset=offset
    )
    return UserList(items=[UserAdminView.model_validate(u) for u in items], total=total)


@router.get("/{user_id}", response_model=UserAdminView, summary="Get one user")
async def get_user(
    user_id: str,
    _current_user: User = require_permissions(Permission.USER_READ),
    user_service: UserService = Depends(get_user_service),
) -> UserAdminView:
    user = await user_service.get_user(user_id)
    return UserAdminView.model_validate(user)


@router.put(
    "/{user_id}/role", response_model=UserAdminView, summary="Change a user's role"
)
async def update_role(
    user_id: str,
    payload: UpdateUserRoleRequest,
    request: Request,
    current_user: User = require_permissions(Permission.USER_MANAGE),
    user_service: UserService = Depends(get_user_service),
) -> UserAdminView:
    user = await user_service.update_role(
        actor=current_user,
        target_user_id=user_id,
        new_role=payload.role,
        ip_address=_client_ip(request),
    )
    return UserAdminView.model_validate(user)


@router.put(
    "/{user_id}/active",
    response_model=UserAdminView,
    summary="Activate or deactivate a user",
)
async def set_active(
    user_id: str,
    payload: SetUserActiveRequest,
    request: Request,
    current_user: User = require_permissions(Permission.USER_MANAGE),
    user_service: UserService = Depends(get_user_service),
) -> UserAdminView:
    user = await user_service.set_active(
        actor=current_user,
        target_user_id=user_id,
        is_active=payload.is_active,
        reason=payload.reason,
        ip_address=_client_ip(request),
    )
    return UserAdminView.model_validate(user)


@router.post(
    "/{user_id}/unlock",
    response_model=UserAdminView,
    summary="Clear a brute-force lockout",
)
async def unlock_user(
    user_id: str,
    payload: UnlockUserRequest,
    request: Request,
    current_user: User = require_permissions(Permission.USER_MANAGE),
    user_service: UserService = Depends(get_user_service),
) -> UserAdminView:
    user = await user_service.unlock_user(
        actor=current_user,
        target_user_id=user_id,
        reason=payload.reason,
        ip_address=_client_ip(request),
    )
    return UserAdminView.model_validate(user)
