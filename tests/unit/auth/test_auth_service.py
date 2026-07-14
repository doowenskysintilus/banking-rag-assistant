from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from backend.core.config import Settings
from backend.core.exceptions import ConflictError, UnauthorizedError
from backend.core.security import decode_token
from backend.core.token_store import get_refresh_session, is_token_revoked
from backend.models.session import RefreshSession
from backend.models.user import UserRole
from backend.schemas.auth import UserLogin, UserRegister
from backend.services.auth_service import AuthService
from tests.unit.auth.conftest import make_oauth_client, make_user_with_password


def _settings(**overrides) -> Settings:
    defaults = dict(
        secret_key="x" * 16, max_failed_login_attempts=3, account_lockout_minutes=15
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestRegister:
    async def test_creates_a_user_with_a_hashed_password(self, db_session):
        service = AuthService(db_session, _settings())
        user = await service.register(
            UserRegister(
                email="new@example.com", full_name="New User", password="password123"
            )
        )

        assert user.email == "new@example.com"
        assert user.hashed_password != "password123"
        assert user.role == UserRole.CUSTOMER_SUPPORT

    async def test_duplicate_email_raises_conflict(self, db_session):
        service = AuthService(db_session, _settings())
        await service.register(
            UserRegister(email="dup@example.com", full_name="A", password="password123")
        )

        with pytest.raises(ConflictError):
            await service.register(
                UserRegister(
                    email="dup@example.com", full_name="B", password="password456"
                )
            )


class TestAuthenticate:
    async def test_correct_credentials_succeed(self, db_session):
        user, password = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())

        authenticated = await service.authenticate(
            UserLogin(email=user.email, password=password)
        )
        assert authenticated.id == user.id

    async def test_unknown_email_raises_unauthorized(self, db_session):
        service = AuthService(db_session, _settings())
        with pytest.raises(UnauthorizedError):
            await service.authenticate(
                UserLogin(email="nobody@example.com", password="whatever1")
            )

    async def test_wrong_password_raises_unauthorized_and_increments_counter(
        self, db_session
    ):
        user, password = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())

        with pytest.raises(UnauthorizedError):
            await service.authenticate(
                UserLogin(email=user.email, password="wrong-password")
            )

        await db_session.refresh(user)
        assert user.failed_login_attempts == 1

    async def test_account_locks_after_max_failed_attempts(self, db_session):
        user, password = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings(max_failed_login_attempts=3))

        for _ in range(3):
            with pytest.raises(UnauthorizedError):
                await service.authenticate(
                    UserLogin(email=user.email, password="wrong-pass")
                )

        await db_session.refresh(user)
        assert user.locked_until is not None

        # Even the correct password is now rejected while locked.
        with pytest.raises(UnauthorizedError):
            await service.authenticate(UserLogin(email=user.email, password=password))

    async def test_successful_login_resets_the_failure_counter(self, db_session):
        user, password = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings(max_failed_login_attempts=5))

        with pytest.raises(UnauthorizedError):
            await service.authenticate(
                UserLogin(email=user.email, password="wrong-pass")
            )
        await service.authenticate(UserLogin(email=user.email, password=password))

        await db_session.refresh(user)
        assert user.failed_login_attempts == 0
        assert user.locked_until is None

    async def test_inactive_account_is_rejected(self, db_session):
        user, password = await make_user_with_password(db_session, is_active=False)
        await db_session.commit()
        service = AuthService(db_session, _settings())

        with pytest.raises(UnauthorizedError):
            await service.authenticate(UserLogin(email=user.email, password=password))


class TestIssueTokens:
    async def test_issues_a_token_pair_with_role_scoped_scope_claim(self, db_session):
        user, _ = await make_user_with_password(db_session, role=UserRole.ANALYST)
        await db_session.commit()
        service = AuthService(db_session, _settings())

        tokens = await service.issue_tokens(user)

        assert tokens.access_token
        assert tokens.refresh_token
        assert "chat:query" in tokens.scope
        assert "user:manage" not in tokens.scope  # ANALYST doesn't hold this permission
        assert tokens.expires_in > 0

        payload = decode_token(tokens.access_token, expected_type="access")
        assert payload["sub"] == user.id
        assert payload["scope"] == tokens.scope

    async def test_creates_a_refresh_session_row(self, db_session):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())

        tokens = await service.issue_tokens(user)
        refresh_payload = decode_token(tokens.refresh_token, expected_type="refresh")

        session = await get_refresh_session(db_session, refresh_payload["jti"])
        assert session is not None
        assert session.user_id == user.id
        assert session.is_active is True


class TestRefresh:
    async def test_rotates_the_refresh_token(self, db_session):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())
        first = await service.issue_tokens(user)

        second = await service.refresh(first.refresh_token)

        assert second.access_token != first.access_token
        assert second.refresh_token != first.refresh_token

        old_jti = decode_token(first.refresh_token, expected_type="refresh")["jti"]
        old_session = await get_refresh_session(db_session, old_jti)
        assert old_session.revoked_at is not None

    async def test_reusing_a_rotated_out_refresh_token_revokes_every_session(
        self, db_session
    ):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())
        first = await service.issue_tokens(user)
        await service.refresh(first.refresh_token)

        with pytest.raises(UnauthorizedError):
            await service.refresh(
                first.refresh_token
            )  # reuse of the now-rotated-away token

        result = await db_session.execute(
            select(RefreshSession).where(RefreshSession.user_id == user.id)
        )
        sessions = result.scalars().all()
        assert all(s.revoked_at is not None for s in sessions)

    async def test_unknown_refresh_token_is_rejected(self, db_session):
        from backend.core.security import create_refresh_token

        service = AuthService(db_session, _settings())
        fake_token = create_refresh_token("some-user-id-with-no-session").token

        with pytest.raises(UnauthorizedError):
            await service.refresh(fake_token)

    async def test_access_token_presented_as_refresh_is_rejected(self, db_session):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())
        tokens = await service.issue_tokens(user)

        with pytest.raises(UnauthorizedError):
            await service.refresh(tokens.access_token)


class TestLogout:
    async def test_revokes_the_access_token_and_refresh_session(self, db_session):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())
        tokens = await service.issue_tokens(user)

        access_payload = decode_token(tokens.access_token, expected_type="access")
        await service.logout(
            user=user,
            access_token_jti=access_payload["jti"],
            access_token_expires_at=datetime.fromtimestamp(
                access_payload["exp"], tz=timezone.utc
            ),
            refresh_token=tokens.refresh_token,
        )

        assert await is_token_revoked(db_session, access_payload["jti"]) is True
        refresh_jti = decode_token(tokens.refresh_token, expected_type="refresh")["jti"]
        session = await get_refresh_session(db_session, refresh_jti)
        assert session.revoked_at is not None


class TestChangePassword:
    async def test_wrong_current_password_is_rejected(self, db_session):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())

        with pytest.raises(UnauthorizedError):
            await service.change_password(
                user=user,
                current_password="wrong-pass",
                new_password="new-password-123",
            )

    async def test_success_rehashes_password_and_revokes_all_sessions(self, db_session):
        user, password = await make_user_with_password(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())
        tokens = await service.issue_tokens(user)

        await service.change_password(
            user=user, current_password=password, new_password="brand-new-password-1"
        )

        refresh_jti = decode_token(tokens.refresh_token, expected_type="refresh")["jti"]
        session = await get_refresh_session(db_session, refresh_jti)
        assert session.revoked_at is not None

        # Old password no longer works; new one does.
        with pytest.raises(UnauthorizedError):
            await service.authenticate(UserLogin(email=user.email, password=password))
        await service.authenticate(
            UserLogin(email=user.email, password="brand-new-password-1")
        )


class TestClientCredentials:
    async def test_valid_client_is_authenticated_with_intersected_scope(
        self, db_session
    ):
        client, secret = await make_oauth_client(
            db_session, allowed_scopes="chat:query document:read"
        )
        await db_session.commit()
        service = AuthService(db_session, _settings())

        authenticated, granted = await service.authenticate_client(
            client.client_id, secret, "chat:query"
        )
        assert authenticated.client_id == client.client_id
        assert granted == "chat:query"

    async def test_requesting_no_scope_grants_everything_allowed(self, db_session):
        client, secret = await make_oauth_client(
            db_session, allowed_scopes="chat:query document:read"
        )
        await db_session.commit()
        service = AuthService(db_session, _settings())

        _, granted = await service.authenticate_client(client.client_id, secret, "")
        assert set(granted.split()) == {"chat:query", "document:read"}

    async def test_requesting_a_scope_outside_allowed_is_not_granted(self, db_session):
        client, secret = await make_oauth_client(
            db_session, allowed_scopes="chat:query"
        )
        await db_session.commit()
        service = AuthService(db_session, _settings())

        _, granted = await service.authenticate_client(
            client.client_id, secret, "chat:query user:manage"
        )
        assert granted == "chat:query"

    async def test_wrong_secret_is_rejected(self, db_session):
        client, _ = await make_oauth_client(db_session)
        await db_session.commit()
        service = AuthService(db_session, _settings())

        with pytest.raises(UnauthorizedError):
            await service.authenticate_client(client.client_id, "wrong-secret", "")

    async def test_unknown_client_id_is_rejected(self, db_session):
        service = AuthService(db_session, _settings())
        with pytest.raises(UnauthorizedError):
            await service.authenticate_client("does-not-exist", "whatever", "")

    async def test_inactive_client_is_rejected(self, db_session):
        client, secret = await make_oauth_client(db_session, is_active=False)
        await db_session.commit()
        service = AuthService(db_session, _settings())

        with pytest.raises(UnauthorizedError):
            await service.authenticate_client(client.client_id, secret, "")

    async def test_issue_client_token_has_no_refresh_token(self, db_session):
        client, secret = await make_oauth_client(
            db_session, allowed_scopes="chat:query"
        )
        await db_session.commit()
        service = AuthService(db_session, _settings())

        _, granted = await service.authenticate_client(
            client.client_id, secret, "chat:query"
        )
        tokens = await service.issue_client_token(client, granted)

        assert tokens.refresh_token is None
        payload = decode_token(tokens.access_token, expected_type="access")
        assert payload["sub"] == f"client:{client.client_id}"
