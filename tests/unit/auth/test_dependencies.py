"""`get_current_user`/`require_permissions` are FastAPI dependencies, but their
logic is plain async code — calling them directly (bypassing FastAPI's DI
wiring) exercises the same code path without needing a full HTTP test client,
which this repo doesn't otherwise use.
"""

import pytest
from fastapi.security import SecurityScopes

from backend.core.dependencies import get_current_user, require_permissions
from backend.core.exceptions import ForbiddenError, UnauthorizedError
from backend.core.permissions import Permission
from backend.core.security import create_access_token
from backend.core.token_store import revoke_access_token
from tests.unit.auth.conftest import make_user_with_password

_NO_SCOPES = SecurityScopes(scopes=[])


def _scopes(*names: str) -> SecurityScopes:
    return SecurityScopes(scopes=list(names))


async def _token_for(user, *, scope: str = "") -> str:
    return create_access_token(
        user.id, extra_claims={"role": user.role.value, "scope": scope}
    ).token


class TestGetCurrentUser:
    async def test_valid_token_with_no_required_scopes_returns_the_user(
        self, db_session
    ):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        token = await _token_for(user)

        result = await get_current_user(_NO_SCOPES, token, db_session)
        assert result.id == user.id

    async def test_missing_token_raises_unauthorized(self, db_session):
        with pytest.raises(UnauthorizedError):
            await get_current_user(_NO_SCOPES, None, db_session)

    async def test_malformed_token_raises_unauthorized(self, db_session):
        with pytest.raises(UnauthorizedError):
            await get_current_user(_NO_SCOPES, "not-a-real-jwt", db_session)

    async def test_token_for_an_unknown_user_id_raises_unauthorized(self, db_session):
        token = create_access_token("no-such-user-id", extra_claims={"scope": ""}).token
        with pytest.raises(UnauthorizedError):
            await get_current_user(_NO_SCOPES, token, db_session)

    async def test_inactive_user_is_rejected(self, db_session):
        user, _ = await make_user_with_password(db_session, is_active=False)
        await db_session.commit()
        token = await _token_for(user)

        with pytest.raises(UnauthorizedError):
            await get_current_user(_NO_SCOPES, token, db_session)

    async def test_revoked_token_is_rejected(self, db_session):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        issued = create_access_token(user.id, extra_claims={"scope": ""})

        await revoke_access_token(
            db_session, jti=issued.jti, expires_at=issued.expires_at, reason="logout"
        )
        await db_session.commit()

        with pytest.raises(UnauthorizedError):
            await get_current_user(_NO_SCOPES, issued.token, db_session)

    async def test_token_with_required_scope_present_succeeds(self, db_session):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        token = await _token_for(user, scope="chat:query document:read")

        result = await get_current_user(_scopes("chat:query"), token, db_session)
        assert result.id == user.id

    async def test_token_missing_a_required_scope_raises_forbidden(self, db_session):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        token = await _token_for(user, scope="chat:query")

        with pytest.raises(ForbiddenError):
            await get_current_user(_scopes("user:manage"), token, db_session)


class TestRequirePermissions:
    def test_returns_a_security_dependency_wrapping_get_current_user(self):
        dependency = require_permissions(Permission.USER_MANAGE)
        assert dependency.dependency is get_current_user
        assert dependency.scopes == ["user:manage"]

    def test_multiple_permissions_all_become_required_scopes(self):
        dependency = require_permissions(Permission.USER_MANAGE, Permission.AUDIT_READ)
        assert set(dependency.scopes) == {"user:manage", "audit:read"}
