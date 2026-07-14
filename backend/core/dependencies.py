"""Central place for FastAPI `Depends()` providers — the app's dependency-injection wiring.

Routers should only ever import from here (plus schemas) — they should never import
services, the LLM client, or the vector store directly. This keeps the DI graph in one
place and makes it trivial to override dependencies in tests (`app.dependency_overrides`).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import Settings, get_settings
from backend.core.database import get_db
from backend.core.exceptions import ForbiddenError, UnauthorizedError
from backend.core.permissions import Permission
from backend.core.security import TokenError, decode_token
from backend.core.token_store import is_token_revoked
from backend.evaluation.service import EvaluationService, get_evaluation_service
from backend.llm.client import OpenAIClient, get_openai_client
from backend.models.user import User
from backend.rag.engine import RAGEngine
from backend.retrieval.embeddings import EmbeddingService, get_embedding_service
from backend.retrieval.reranking.service import RerankingService, get_reranking_service
from backend.retrieval.vector_store import VectorStoreBase, get_vector_store
from backend.services.audit_service import AuditService
from backend.services.auth_service import AuthService
from backend.services.chat_service import ChatService
from backend.services.document_service import DocumentService
from backend.services.user_service import UserService

# --- Infrastructure singletons -------------------------------------------------

SettingsDep = Depends(get_settings)


def get_openai_client_dep() -> OpenAIClient:
    return get_openai_client()


def get_embedding_service_dep() -> EmbeddingService:
    return get_embedding_service()


def get_vector_store_dep() -> VectorStoreBase:
    return get_vector_store()


def get_reranking_service_dep() -> RerankingService:
    return get_reranking_service()


def get_evaluation_service_dep() -> EvaluationService:
    return get_evaluation_service()


def get_rag_engine(
    settings: Settings = SettingsDep,
    openai_client: OpenAIClient = Depends(get_openai_client_dep),
    embedding_service: EmbeddingService = Depends(get_embedding_service_dep),
    vector_store: VectorStoreBase = Depends(get_vector_store_dep),
    reranking_service: RerankingService = Depends(get_reranking_service_dep),
) -> RAGEngine:
    enterprise_retrieval = None
    if settings.rag_retrieval_engine == "enterprise":
        # Imported lazily so the "native" engine never pulls in the LangChain
        # dependency stack — see the TYPE_CHECKING import note in rag/engine.py.
        from backend.retrieval.enterprise import EnterpriseRetrievalService

        enterprise_retrieval = EnterpriseRetrievalService(
            embedding_service, vector_store, settings
        )

    return RAGEngine(
        embedding_service=embedding_service,
        vector_store=vector_store,
        reranking_service=reranking_service,
        openai_client=openai_client,
        settings=settings,
        enterprise_retrieval=enterprise_retrieval,
    )


# --- Authentication (OAuth2 + RBAC) -----------------------------------------------
#
# `oauth2_scheme` is what makes this a real OAuth2 API rather than "a bearer
# token that happens to be a JWT": it advertises the token endpoint and scope
# vocabulary to Swagger's "Authorize" button (`/docs`), and every
# `backend.core.permissions.Permission` doubles as a scope name here — a
# token's `scope` claim (set by `AuthService.issue_tokens`/`authenticate_client`
# to the caller's full role/client grant) is checked against whatever scopes an
# endpoint declares via `Security(get_current_user, scopes=[...])` (see
# `require_permissions` below). `auto_error=False` so a missing token raises
# this app's own `UnauthorizedError` (consistent JSON envelope) instead of
# FastAPI's default.

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=get_settings().oauth2_token_url,
    scopes={p.value: p.value for p in Permission},
    auto_error=False,
)


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if token is None:
        raise UnauthorizedError("Missing bearer token")

    try:
        payload = decode_token(token, expected_type="access")
    except TokenError as exc:
        raise UnauthorizedError(str(exc)) from exc

    jti = payload.get("jti")
    if jti and await is_token_revoked(db, jti):
        raise UnauthorizedError("This token has been revoked")

    user_id = payload.get("sub")
    user = await db.get(User, user_id) if user_id else None
    if user is None or not user.is_active:
        raise UnauthorizedError("User not found or inactive")

    if security_scopes.scopes:
        token_scopes = set(payload.get("scope", "").split())
        missing = [s for s in security_scopes.scopes if s not in token_scopes]
        if missing:
            raise ForbiddenError(
                f"Token is missing required scope(s): {', '.join(missing)}"
            )

    return user


def require_permissions(*permissions: Permission):
    """`current_user: User = require_permissions(Permission.USER_MANAGE)` — this
    already returns a `Security(...)` params marker (like `Depends(...)`), so
    use it directly as the parameter default, not wrapped in `Depends(...)`.
    Requires the presented token's `scope` claim to carry every listed
    permission. Since
    `AuthService` always issues human users a token scoped to their role's
    full permission set (see `backend.core.permissions.scopes_for_role`), this
    is RBAC (what the user's role allows) and OAuth2 scope enforcement (what
    this particular token was granted) expressed as one check — the two only
    diverge for client-credentials tokens, which can be issued a deliberately
    narrower scope than a role would carry (see `OAuthClient.allowed_scopes`).
    """
    return Security(get_current_user, scopes=[p.value for p in permissions])


# --- Services ----------------------------------------------------------------------


def get_auth_service(
    db: AsyncSession = Depends(get_db),
    settings: Settings = SettingsDep,
) -> AuthService:
    return AuthService(db=db, settings=settings)


def get_chat_service(
    db: AsyncSession = Depends(get_db),
    rag_engine: RAGEngine = Depends(get_rag_engine),
) -> ChatService:
    """Full service for /chat/query and /chat/stream — requires the RAG engine."""
    return ChatService(db=db, rag_engine=rag_engine)


def get_conversation_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    """Lightweight service for conversation history endpoints — pure DB access, so it
    avoids initializing the embedding model and vector store on every call.
    """
    return ChatService(db=db, rag_engine=None)


def get_document_service(
    db: AsyncSession = Depends(get_db),
    settings: Settings = SettingsDep,
) -> DocumentService:
    return DocumentService(db=db, settings=settings)


def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(db=db)


def get_audit_service(db: AsyncSession = Depends(get_db)) -> AuditService:
    return AuditService(db=db)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db():
        yield session
