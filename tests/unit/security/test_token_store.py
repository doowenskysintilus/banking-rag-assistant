import uuid
from datetime import datetime, timedelta, timezone

from backend.core.token_store import (
    create_refresh_session,
    get_refresh_session,
    is_token_revoked,
    purge_expired_revocations,
    revoke_access_token,
    revoke_all_refresh_sessions_for_user,
    revoke_refresh_session,
)
from tests.unit.security.conftest import make_user


def _future(minutes: int = 30) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


def _past(minutes: int = 30) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


class TestRefreshSessions:
    async def test_created_session_is_retrievable_and_active(self, db_session):
        user = await make_user(db_session)
        jti = uuid.uuid4().hex
        await create_refresh_session(
            db_session, jti=jti, user_id=user.id, expires_at=_future()
        )
        await db_session.commit()

        session = await get_refresh_session(db_session, jti)
        assert session is not None
        assert session.user_id == user.id
        assert session.is_active is True

    async def test_revoking_a_session_marks_it_inactive_and_records_replacement(
        self, db_session
    ):
        user = await make_user(db_session)
        jti, new_jti = uuid.uuid4().hex, uuid.uuid4().hex
        await create_refresh_session(
            db_session, jti=jti, user_id=user.id, expires_at=_future()
        )
        await db_session.commit()

        await revoke_refresh_session(db_session, jti, replaced_by_jti=new_jti)
        await db_session.commit()

        session = await get_refresh_session(db_session, jti)
        assert session.revoked_at is not None
        assert session.replaced_by_jti == new_jti
        assert session.is_active is False

    async def test_revoking_an_already_revoked_session_is_a_no_op(self, db_session):
        user = await make_user(db_session)
        jti = uuid.uuid4().hex
        await create_refresh_session(
            db_session, jti=jti, user_id=user.id, expires_at=_future()
        )
        await db_session.commit()

        await revoke_refresh_session(db_session, jti)
        await db_session.commit()
        first_revoked_at = (await get_refresh_session(db_session, jti)).revoked_at

        await revoke_refresh_session(
            db_session, jti, replaced_by_jti="should-not-apply"
        )
        await db_session.commit()
        session = await get_refresh_session(db_session, jti)

        assert session.revoked_at == first_revoked_at
        assert session.replaced_by_jti is None

    async def test_expired_but_not_revoked_session_is_not_active(self, db_session):
        user = await make_user(db_session)
        jti = uuid.uuid4().hex
        await create_refresh_session(
            db_session, jti=jti, user_id=user.id, expires_at=_past()
        )
        await db_session.commit()

        session = await get_refresh_session(db_session, jti)
        assert session.is_active is False

    async def test_revoke_all_for_user_only_touches_that_users_active_sessions(
        self, db_session
    ):
        user_a = await make_user(db_session)
        user_b = await make_user(db_session)
        jti_a1, jti_a2, jti_b1 = uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex
        await create_refresh_session(
            db_session, jti=jti_a1, user_id=user_a.id, expires_at=_future()
        )
        await create_refresh_session(
            db_session, jti=jti_a2, user_id=user_a.id, expires_at=_future()
        )
        await create_refresh_session(
            db_session, jti=jti_b1, user_id=user_b.id, expires_at=_future()
        )
        await db_session.commit()

        revoked_count = await revoke_all_refresh_sessions_for_user(
            db_session, user_a.id
        )
        await db_session.commit()

        assert revoked_count == 2
        assert (await get_refresh_session(db_session, jti_a1)).revoked_at is not None
        assert (await get_refresh_session(db_session, jti_a2)).revoked_at is not None
        assert (await get_refresh_session(db_session, jti_b1)).revoked_at is None

    async def test_revoke_all_does_not_touch_already_revoked_sessions_twice(
        self, db_session
    ):
        user = await make_user(db_session)
        jti = uuid.uuid4().hex
        await create_refresh_session(
            db_session, jti=jti, user_id=user.id, expires_at=_future()
        )
        await db_session.commit()
        await revoke_refresh_session(db_session, jti)
        await db_session.commit()

        revoked_count = await revoke_all_refresh_sessions_for_user(db_session, user.id)
        assert revoked_count == 0


class TestRevokedTokenDenylist:
    async def test_unrevoked_token_is_not_reported_as_revoked(self, db_session):
        assert await is_token_revoked(db_session, uuid.uuid4().hex) is False

    async def test_revoked_token_is_reported_as_revoked(self, db_session):
        jti = uuid.uuid4().hex
        await revoke_access_token(
            db_session, jti=jti, expires_at=_future(), reason="logout"
        )
        await db_session.commit()

        assert await is_token_revoked(db_session, jti) is True

    async def test_purge_removes_only_expired_entries(self, db_session):
        expired_jti, live_jti = uuid.uuid4().hex, uuid.uuid4().hex
        await revoke_access_token(
            db_session, jti=expired_jti, expires_at=_past(), reason="logout"
        )
        await revoke_access_token(
            db_session, jti=live_jti, expires_at=_future(), reason="logout"
        )
        await db_session.commit()

        purged = await purge_expired_revocations(db_session)
        await db_session.commit()

        assert purged == 1
        assert await is_token_revoked(db_session, expired_jti) is False
        assert await is_token_revoked(db_session, live_jti) is True
