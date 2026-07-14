"""Registered machine-to-machine clients for the OAuth2 client-credentials
grant (see `backend.services.auth_service.AuthService.authenticate_client` and
`POST /auth/token` with `grant_type=client_credentials`) — for service
integrations (batch jobs, other internal systems) that aren't a human sitting
behind a password login.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class OAuthClient(Base):
    __tablename__ = "oauth_clients"

    client_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    hashed_secret: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Space-separated backend.core.permissions.Permission values this client
    # may ever request — a token's actual `scope` claim is the intersection of
    # this and whatever the token request asked for (see AuthService).
    allowed_scopes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
