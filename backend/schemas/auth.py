from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from backend.models.user import UserRole
from backend.schemas.common import ORMModel


class UserRegister(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserPublic(ORMModel):
    id: str
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int = Field(description="Seconds until the access token expires.")
    scope: str = Field(
        default="", description="Space-separated OAuth2 scopes this token was granted."
    )


class RefreshRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)
