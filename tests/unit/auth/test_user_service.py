from datetime import datetime, timedelta, timezone

import pytest

from backend.core.exceptions import ForbiddenError, NotFoundError
from backend.models.user import UserRole
from backend.services.user_service import UserService
from tests.unit.auth.conftest import make_user_with_password


class TestListUsers:
    async def test_lists_all_users_with_total_count(self, db_session):
        await make_user_with_password(db_session, role=UserRole.ADMIN)
        await make_user_with_password(db_session, role=UserRole.ANALYST)
        await db_session.commit()
        service = UserService(db_session)

        items, total = await service.list_users()
        assert total == 2
        assert len(items) == 2

    async def test_filters_by_role(self, db_session):
        await make_user_with_password(db_session, role=UserRole.ADMIN)
        await make_user_with_password(db_session, role=UserRole.ANALYST)
        await db_session.commit()
        service = UserService(db_session)

        items, total = await service.list_users(role=UserRole.ADMIN)
        assert total == 1
        assert items[0].role == UserRole.ADMIN

    async def test_filters_by_active_state(self, db_session):
        await make_user_with_password(db_session, is_active=True)
        await make_user_with_password(db_session, is_active=False)
        await db_session.commit()
        service = UserService(db_session)

        items, total = await service.list_users(is_active=False)
        assert total == 1
        assert items[0].is_active is False

    async def test_respects_limit_and_offset(self, db_session):
        for _ in range(5):
            await make_user_with_password(db_session)
        await db_session.commit()
        service = UserService(db_session)

        items, total = await service.list_users(limit=2, offset=0)
        assert total == 5
        assert len(items) == 2


class TestGetUser:
    async def test_returns_the_user(self, db_session):
        user, _ = await make_user_with_password(db_session)
        await db_session.commit()
        service = UserService(db_session)

        found = await service.get_user(user.id)
        assert found.id == user.id

    async def test_unknown_id_raises_not_found(self, db_session):
        service = UserService(db_session)
        with pytest.raises(NotFoundError):
            await service.get_user("does-not-exist")


class TestUpdateRole:
    async def test_changes_the_target_users_role(self, db_session):
        admin, _ = await make_user_with_password(db_session, role=UserRole.ADMIN)
        target, _ = await make_user_with_password(
            db_session, role=UserRole.CUSTOMER_SUPPORT
        )
        await db_session.commit()
        service = UserService(db_session)

        updated = await service.update_role(
            actor=admin, target_user_id=target.id, new_role=UserRole.ANALYST
        )
        assert updated.role == UserRole.ANALYST

    async def test_cannot_change_own_role(self, db_session):
        admin, _ = await make_user_with_password(db_session, role=UserRole.ADMIN)
        await db_session.commit()
        service = UserService(db_session)

        with pytest.raises(ForbiddenError):
            await service.update_role(
                actor=admin, target_user_id=admin.id, new_role=UserRole.ANALYST
            )


class TestSetActive:
    async def test_deactivates_the_target_user(self, db_session):
        admin, _ = await make_user_with_password(db_session, role=UserRole.ADMIN)
        target, _ = await make_user_with_password(db_session, is_active=True)
        await db_session.commit()
        service = UserService(db_session)

        updated = await service.set_active(
            actor=admin,
            target_user_id=target.id,
            is_active=False,
            reason="fraud review",
        )
        assert updated.is_active is False

    async def test_cannot_deactivate_self(self, db_session):
        admin, _ = await make_user_with_password(db_session, role=UserRole.ADMIN)
        await db_session.commit()
        service = UserService(db_session)

        with pytest.raises(ForbiddenError):
            await service.set_active(
                actor=admin, target_user_id=admin.id, is_active=False
            )


class TestUnlockUser:
    async def test_clears_lockout_state(self, db_session):
        admin, _ = await make_user_with_password(db_session, role=UserRole.ADMIN)
        target, _ = await make_user_with_password(db_session)
        target.failed_login_attempts = 5
        target.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        await db_session.commit()
        service = UserService(db_session)

        updated = await service.unlock_user(
            actor=admin, target_user_id=target.id, reason="verified via phone"
        )

        assert updated.failed_login_attempts == 0
        assert updated.locked_until is None


class TestIsLocked:
    def test_true_when_locked_until_in_the_future(self):
        from backend.models.user import User

        user = User(
            email="a@b.com",
            full_name="A",
            hashed_password="x",
            locked_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        assert UserService.is_locked(user) is True

    def test_false_when_locked_until_in_the_past(self):
        from backend.models.user import User

        user = User(
            email="a@b.com",
            full_name="A",
            hashed_password="x",
            locked_until=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        assert UserService.is_locked(user) is False

    def test_false_when_never_locked(self):
        from backend.models.user import User

        user = User(email="a@b.com", full_name="A", hashed_password="x")
        assert UserService.is_locked(user) is False
