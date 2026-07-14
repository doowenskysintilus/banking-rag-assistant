"""Fine-grained RBAC on top of `backend.models.user.UserRole`.

Three roles are enough to decide *who a user is*, but not always enough to
decide *what a specific endpoint requires* — `Permission` is the finer unit
access checks are actually expressed in, and `ROLE_PERMISSIONS` is the single
place that maps roles to the set of permissions they carry. Adding a role or
changing what a role can do means editing this file only, not hunting through
routers for scattered `if role == ...` checks.

Each `Permission` value doubles as an OAuth2 scope string (see
`backend.core.dependencies`'s `oauth2_scheme` and `backend.api.v1.routers.auth`'s
`/auth/token`) — one vocabulary for "what can this role do" and "what did this
token's owner consent to", rather than maintaining two parallel lists.
"""

from __future__ import annotations

import enum

from backend.models.user import UserRole


class Permission(str, enum.Enum):
    CHAT_QUERY = "chat:query"
    CONVERSATION_READ = "conversation:read"

    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"

    EVALUATION_READ = "evaluation:read"
    EVALUATION_RUN = "evaluation:run"

    USER_READ = "user:read"
    USER_MANAGE = "user:manage"

    AUDIT_READ = "audit:read"


# Every permission a role holds is granted implicitly, without needing an
# explicit per-endpoint role list — `require_permissions` (see
# backend.core.dependencies) is what routers should depend on, not `UserRole`
# directly, so this table stays the single source of truth.
ROLE_PERMISSIONS: dict[UserRole, frozenset[Permission]] = {
    UserRole.ADMIN: frozenset(Permission),  # admins hold every permission
    UserRole.ANALYST: frozenset(
        {
            Permission.CHAT_QUERY,
            Permission.CONVERSATION_READ,
            Permission.DOCUMENT_READ,
            Permission.DOCUMENT_WRITE,
            Permission.EVALUATION_READ,
            Permission.EVALUATION_RUN,
        }
    ),
    UserRole.CUSTOMER_SUPPORT: frozenset(
        {
            Permission.CHAT_QUERY,
            Permission.CONVERSATION_READ,
            Permission.DOCUMENT_READ,
        }
    ),
}


def permissions_for_role(role: UserRole) -> frozenset[Permission]:
    return ROLE_PERMISSIONS.get(role, frozenset())


def scopes_for_role(role: UserRole) -> list[str]:
    """OAuth2 scope strings a token issued to this role may carry."""
    return sorted(p.value for p in permissions_for_role(role))
