"""User registration, authentication, and OAuth2 token issuance/lifecycle.

Covers the full "enterprise banking" JWT story, not just sign-and-verify:
- Brute-force lockout (`authenticate`): a bank cannot allow unlimited password
  guesses against a customer-facing login.
- Refresh-token rotation with reuse detection (`refresh`): a stolen refresh
  token used after the legitimate client already rotated past it is a strong
  signal of theft — the response is to revoke every session for that user,
  not just the one token.
- Explicit revocation (`logout`, `change_password`): a bare JWT can't be
  "un-issued", so both write to `backend.core.token_store`'s denylist/session
  tables rather than relying on the token's own expiry.
- OAuth2 client-credentials grant (`authenticate_client`) for machine clients
  alongside the password grant for human users — see
  `backend.models.oauth_client.OAuthClient`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.audit import append_audit_log
from backend.core.config import Settings
from backend.core.datetime_utils import ensure_utc
from backend.core.exceptions import ConflictError, UnauthorizedError
from backend.core.logging import get_logger
from backend.core.permissions import scopes_for_role
from backend.core.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from backend.core.token_store import (
    create_refresh_session,
    get_refresh_session,
    revoke_access_token,
    revoke_all_refresh_sessions_for_user,
    revoke_refresh_session,
)
from backend.models.oauth_client import OAuthClient
from backend.models.user import User
from backend.schemas.auth import TokenPair, UserLogin, UserRegister

logger = get_logger(__name__)


class AuthService:
    def __init__(self, db: AsyncSession, settings: Settings) -> None:
        self._db = db
        self._settings = settings

    # --- Registration ---------------------------------------------------------

    async def register(
        self, payload: UserRegister, *, ip_address: str | None = None
    ) -> User:
        existing = await self._db.execute(
            select(User).where(User.email == payload.email)
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError("An account with this email already exists")

        user = User(
            email=payload.email,
            full_name=payload.full_name,
            hashed_password=hash_password(payload.password),
        )
        self._db.add(user)
        await self._db.flush()
        await append_audit_log(
            self._db,
            user_id=user.id,
            action="user_registered",
            resource=user.id,
            ip_address=ip_address,
        )
        await self._db.commit()
        await self._db.refresh(user)
        logger.info("user_registered", extra={"user_id": user.id})
        return user

    # --- Authentication (password grant) --------------------------------------

    async def authenticate(
        self, payload: UserLogin, *, ip_address: str | None = None
    ) -> User:
        result = await self._db.execute(select(User).where(User.email == payload.email))
        user = result.scalar_one_or_none()

        if user is None:
            # No user_id to attach — logged without one rather than guessing at
            # who was targeted, which would mean writing an attacker-controlled
            # email into the audit trail.
            await append_audit_log(
                self._db,
                user_id=None,
                action="login_failed_unknown_email",
                ip_address=ip_address,
            )
            await self._db.commit()
            raise UnauthorizedError("Invalid email or password")

        if self._is_locked(user):
            await append_audit_log(
                self._db,
                user_id=user.id,
                action="login_blocked_locked",
                resource=user.id,
                ip_address=ip_address,
            )
            await self._db.commit()
            raise UnauthorizedError(
                "This account is temporarily locked due to repeated failed login attempts"
            )

        if not user.is_active:
            await append_audit_log(
                self._db,
                user_id=user.id,
                action="login_blocked_inactive",
                resource=user.id,
                ip_address=ip_address,
            )
            await self._db.commit()
            raise UnauthorizedError("This account has been deactivated")

        if not verify_password(payload.password, user.hashed_password):
            await self._register_failed_attempt(user, ip_address=ip_address)
            raise UnauthorizedError("Invalid email or password")

        user.failed_login_attempts = 0
        user.locked_until = None
        await append_audit_log(
            self._db,
            user_id=user.id,
            action="login_success",
            resource=user.id,
            ip_address=ip_address,
        )
        await self._db.commit()
        await self._db.refresh(user)
        return user

    async def _register_failed_attempt(
        self, user: User, *, ip_address: str | None
    ) -> None:
        user.failed_login_attempts += 1
        locked = user.failed_login_attempts >= self._settings.max_failed_login_attempts
        if locked:
            user.locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=self._settings.account_lockout_minutes
            )
        await append_audit_log(
            self._db,
            user_id=user.id,
            action="account_locked" if locked else "login_failed",
            resource=user.id,
            detail=f"failed_attempts={user.failed_login_attempts}",
            ip_address=ip_address,
        )
        await self._db.commit()

    @staticmethod
    def _is_locked(user: User) -> bool:
        return user.locked_until is not None and ensure_utc(
            user.locked_until
        ) > datetime.now(timezone.utc)

    # --- Token issuance/lifecycle ----------------------------------------------

    async def issue_tokens(
        self,
        user: User,
        *,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> TokenPair:
        scope = " ".join(scopes_for_role(user.role))
        access = create_access_token(
            user.id, extra_claims={"role": user.role.value, "scope": scope}
        )
        refresh = create_refresh_token(user.id)

        await create_refresh_session(
            self._db,
            jti=refresh.jti,
            user_id=user.id,
            expires_at=refresh.expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        await self._db.commit()

        return TokenPair(
            access_token=access.token,
            refresh_token=refresh.token,
            expires_in=int(
                (access.expires_at - datetime.now(timezone.utc)).total_seconds()
            ),
            scope=scope,
        )

    async def refresh(
        self,
        refresh_token: str,
        *,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> TokenPair:
        try:
            payload = decode_token(refresh_token, expected_type="refresh")
        except TokenError as exc:
            raise UnauthorizedError(str(exc)) from exc

        jti = payload.get("jti")
        user_id = payload.get("sub")
        session = await get_refresh_session(self._db, jti) if jti else None

        if session is None or user_id is None:
            raise UnauthorizedError("Refresh token is invalid or unknown")

        if session.revoked_at is not None:
            # Reuse of an already-rotated-away (or already-logged-out) refresh
            # token — treat as a possible theft: kill every session this user
            # has, not just this one, and force re-authentication everywhere.
            await revoke_all_refresh_sessions_for_user(self._db, user_id)
            await append_audit_log(
                self._db,
                user_id=user_id,
                action="refresh_token_reuse_detected",
                detail="all sessions revoked",
                ip_address=ip_address,
            )
            await self._db.commit()
            raise UnauthorizedError(
                "This refresh token has already been used — all sessions have been revoked for safety"
            )

        if ensure_utc(session.expires_at) <= datetime.now(timezone.utc):
            raise UnauthorizedError("Refresh token has expired")

        user = await self._db.get(User, user_id)
        if user is None or not user.is_active:
            raise UnauthorizedError("User not found or inactive")

        # Rotate: the presented refresh token is consumed exactly once.
        scope = " ".join(scopes_for_role(user.role))
        access = create_access_token(
            user.id, extra_claims={"role": user.role.value, "scope": scope}
        )
        new_refresh = create_refresh_token(user.id)

        await revoke_refresh_session(self._db, jti, replaced_by_jti=new_refresh.jti)
        await create_refresh_session(
            self._db,
            jti=new_refresh.jti,
            user_id=user.id,
            expires_at=new_refresh.expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        await append_audit_log(
            self._db,
            user_id=user.id,
            action="token_refreshed",
            resource=user.id,
            ip_address=ip_address,
        )
        await self._db.commit()

        return TokenPair(
            access_token=access.token,
            refresh_token=new_refresh.token,
            expires_in=int(
                (access.expires_at - datetime.now(timezone.utc)).total_seconds()
            ),
            scope=scope,
        )

    async def logout(
        self,
        *,
        user: User,
        access_token_jti: str,
        access_token_expires_at: datetime,
        refresh_token: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        await revoke_access_token(
            self._db,
            jti=access_token_jti,
            expires_at=access_token_expires_at,
            reason="logout",
        )

        if refresh_token:
            try:
                payload = decode_token(refresh_token, expected_type="refresh")
            except TokenError:
                payload = {}
            refresh_jti = payload.get("jti")
            if refresh_jti:
                await revoke_refresh_session(self._db, refresh_jti)

        await append_audit_log(
            self._db,
            user_id=user.id,
            action="logout",
            resource=user.id,
            ip_address=ip_address,
        )
        await self._db.commit()

    async def change_password(
        self,
        *,
        user: User,
        current_password: str,
        new_password: str,
        ip_address: str | None = None,
    ) -> None:
        if not verify_password(current_password, user.hashed_password):
            raise UnauthorizedError("Current password is incorrect")

        user.hashed_password = hash_password(new_password)
        revoked_count = await revoke_all_refresh_sessions_for_user(self._db, user.id)
        await append_audit_log(
            self._db,
            user_id=user.id,
            action="password_changed",
            resource=user.id,
            detail=f"sessions_revoked={revoked_count}",
            ip_address=ip_address,
        )
        await self._db.commit()
        logger.info(
            "password_changed",
            extra={"user_id": user.id, "sessions_revoked": revoked_count},
        )

    # --- OAuth2 client-credentials grant (machine clients) ---------------------

    async def authenticate_client(
        self, client_id: str, client_secret: str, requested_scope: str
    ) -> tuple[OAuthClient, str]:
        """Returns `(client, granted_scope)` — `granted_scope` is the
        intersection of what was requested and what `client.allowed_scopes`
        actually permits (never wider than either).
        """
        client = await self._db.get(OAuthClient, client_id)
        if (
            client is None
            or not client.is_active
            or not verify_password(client_secret, client.hashed_secret)
        ):
            await append_audit_log(
                self._db, user_id=None, action="client_auth_failed", resource=client_id
            )
            await self._db.commit()
            raise UnauthorizedError("Invalid client credentials")

        allowed = set(client.allowed_scopes.split())
        requested = set(requested_scope.split()) if requested_scope else allowed
        granted = sorted(requested & allowed)

        await append_audit_log(
            self._db, user_id=None, action="client_authenticated", resource=client_id
        )
        await self._db.commit()
        return client, " ".join(granted)

    async def issue_client_token(
        self, client: OAuthClient, granted_scope: str
    ) -> TokenPair:
        access = create_access_token(
            f"client:{client.client_id}",
            extra_claims={"scope": granted_scope, "client_id": client.client_id},
        )
        return TokenPair(
            access_token=access.token,
            refresh_token=None,
            expires_in=int(
                (access.expires_at - datetime.now(timezone.utc)).total_seconds()
            ),
            scope=granted_scope,
        )
