"""Authentication: registration, login (JSON convenience + OAuth2-standard
/token), refresh, logout, and self-service password change.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, Request, status

from backend.core.dependencies import get_auth_service, get_current_user, oauth2_scheme
from backend.core.exceptions import ValidationAppError
from backend.core.security import TokenError, decode_token
from backend.models.user import User
from backend.schemas.auth import (
    ChangePasswordRequest,
    RefreshRequest,
    TokenPair,
    UserLogin,
    UserPublic,
    UserRegister,
)
from backend.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


@router.post(
    "/register",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account",
)
async def register(
    payload: UserRegister,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserPublic:
    user = await auth_service.register(payload, ip_address=_client_ip(request))
    return UserPublic.model_validate(user)


@router.post(
    "/login",
    response_model=TokenPair,
    summary="Exchange credentials (JSON) for a token pair",
)
async def login(
    payload: UserLogin,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenPair:
    ip = _client_ip(request)
    user = await auth_service.authenticate(payload, ip_address=ip)
    return await auth_service.issue_tokens(
        user, ip_address=ip, user_agent=request.headers.get("user-agent")
    )


@router.post(
    "/token",
    response_model=TokenPair,
    summary="OAuth2 token endpoint — grant_type=password (human users) or client_credentials (service clients)",
)
async def token(
    request: Request,
    grant_type: str = Form(...),
    username: str | None = Form(default=None),
    password: str | None = Form(default=None),
    client_id: str | None = Form(default=None),
    client_secret: str | None = Form(default=None),
    scope: str = Form(default=""),
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenPair:
    ip = _client_ip(request)
    user_agent = request.headers.get("user-agent")

    if grant_type == "password":
        if not username or not password:
            raise ValidationAppError(
                "'username' and 'password' are required for grant_type=password"
            )
        user = await auth_service.authenticate(
            UserLogin(email=username, password=password), ip_address=ip
        )
        return await auth_service.issue_tokens(
            user, ip_address=ip, user_agent=user_agent
        )

    if grant_type == "client_credentials":
        if not client_id or not client_secret:
            raise ValidationAppError(
                "'client_id' and 'client_secret' are required for grant_type=client_credentials"
            )
        client, granted_scope = await auth_service.authenticate_client(
            client_id, client_secret, scope
        )
        return await auth_service.issue_client_token(client, granted_scope)

    raise ValidationAppError(
        f"Unsupported grant_type: {grant_type!r} (expected 'password' or 'client_credentials')"
    )


@router.post(
    "/refresh",
    response_model=TokenPair,
    summary="Exchange a refresh token for a new token pair",
)
async def refresh(
    payload: RefreshRequest,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenPair:
    return await auth_service.refresh(
        payload.refresh_token,
        ip_address=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke the current access token and (optionally) a refresh token",
)
async def logout(
    request: Request,
    payload: RefreshRequest | None = None,
    access_token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
) -> None:
    try:
        access_payload = decode_token(access_token, expected_type="access")
    except TokenError:
        # get_current_user already validated this token to resolve current_user
        # above, so this can only fail if it expired in the few ms in between —
        # nothing left to revoke in that case.
        return None

    # jose's jwt.decode returns `exp` as a numeric (Unix timestamp) claim, per
    # the JWT spec — not the `datetime` that was originally passed to encode it.
    expires_at = datetime.fromtimestamp(access_payload["exp"], tz=timezone.utc)

    await auth_service.logout(
        user=current_user,
        access_token_jti=access_payload["jti"],
        access_token_expires_at=expires_at,
        refresh_token=payload.refresh_token if payload else None,
        ip_address=_client_ip(request),
    )


@router.post(
    "/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change the current user's password",
)
async def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
) -> None:
    await auth_service.change_password(
        user=current_user,
        current_password=payload.current_password,
        new_password=payload.new_password,
        ip_address=_client_ip(request),
    )


@router.get(
    "/me", response_model=UserPublic, summary="Return the authenticated user's profile"
)
async def me(current_user: User = Depends(get_current_user)) -> UserPublic:
    return UserPublic.model_validate(current_user)
