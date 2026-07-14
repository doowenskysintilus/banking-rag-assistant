from backend.core.permissions import (
    Permission,
    ROLE_PERMISSIONS,
    permissions_for_role,
    scopes_for_role,
)
from backend.models.user import UserRole


def test_admin_holds_every_permission():
    assert permissions_for_role(UserRole.ADMIN) == frozenset(Permission)


def test_customer_support_cannot_manage_users_or_read_audit():
    perms = permissions_for_role(UserRole.CUSTOMER_SUPPORT)
    assert Permission.USER_MANAGE not in perms
    assert Permission.AUDIT_READ not in perms
    assert Permission.CHAT_QUERY in perms


def test_analyst_can_run_and_read_evaluations():
    perms = permissions_for_role(UserRole.ANALYST)
    assert Permission.EVALUATION_RUN in perms
    assert Permission.EVALUATION_READ in perms
    assert Permission.USER_MANAGE not in perms


def test_every_role_in_the_enum_has_a_permission_set_defined():
    for role in UserRole:
        assert role in ROLE_PERMISSIONS


def test_scopes_for_role_are_sorted_scope_strings():
    scopes = scopes_for_role(UserRole.CUSTOMER_SUPPORT)
    assert scopes == sorted(scopes)
    assert "chat:query" in scopes


def test_permission_values_are_stable_scope_strings():
    # These strings are persisted in issued JWTs' `scope` claim — renaming a
    # Permission's value would silently invalidate every outstanding token's
    # scope check, so pin the exact vocabulary here.
    assert Permission.CHAT_QUERY.value == "chat:query"
    assert Permission.USER_MANAGE.value == "user:manage"
    assert Permission.AUDIT_READ.value == "audit:read"
