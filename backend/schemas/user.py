from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from backend.models.user import UserRole
from backend.schemas.common import ORMModel


class UserAdminView(ORMModel):
    """Fuller user projection for admin management — `UserPublic`
    (backend.schemas.auth) is what a user sees of *themselves*; this adds the
    account-security fields only an admin needs (lockout state).
    """

    id: str
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool
    failed_login_attempts: int
    locked_until: datetime | None
    created_at: datetime
    updated_at: datetime


class UserList(BaseModel):
    items: list[UserAdminView]
    total: int


class UpdateUserRoleRequest(BaseModel):
    role: UserRole


class SetUserActiveRequest(BaseModel):
    is_active: bool
    reason: str | None = Field(default=None, max_length=500)


class UnlockUserRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=500)
