"""DB fixture + user/client seeding helpers for auth/user-management tests —
reuses the enterprise retrieval tests' in-memory SQLite fixture, adding a
helper that hashes a real password (the shared `make_user` sets a fake
`hashed_password="hashed"`, fine for tests that never call `verify_password`,
but useless for anything exercising login).
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.security import hash_password
from backend.models.oauth_client import OAuthClient
from backend.models.user import User, UserRole
from tests.unit.retrieval.enterprise.conftest import db_session

__all__ = ["db_session", "make_user_with_password", "make_oauth_client"]


async def make_user_with_password(
    db: AsyncSession,
    *,
    email: str | None = None,
    password: str = "correct-horse-battery-staple",
    role: UserRole = UserRole.CUSTOMER_SUPPORT,
    is_active: bool = True,
) -> tuple[User, str]:
    """Returns `(user, plaintext_password)` so a test can call
    `AuthService.authenticate` with a real, verifiable password.
    """
    user = User(
        id=str(uuid.uuid4()),
        email=email or f"{uuid.uuid4().hex}@example.com",
        full_name="Test User",
        hashed_password=hash_password(password),
        role=role,
        is_active=is_active,
    )
    db.add(user)
    await db.flush()
    return user, password


async def make_oauth_client(
    db: AsyncSession,
    *,
    client_id: str | None = None,
    secret: str = "s3cr3t",
    allowed_scopes: str = "chat:query",
    is_active: bool = True,
) -> tuple[OAuthClient, str]:
    client = OAuthClient(
        client_id=client_id or f"client-{uuid.uuid4().hex[:8]}",
        hashed_secret=hash_password(secret),
        name="Test Client",
        allowed_scopes=allowed_scopes,
        is_active=is_active,
    )
    db.add(client)
    await db.flush()
    return client, secret
