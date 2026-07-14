"""Persists one audit row per RAG query — success or failure — plus a matching
entry in the app's general compliance trail (`backend.models.audit_log.AuditLog`).

Takes plain values rather than the engine's dataclasses so it has no dependency on
`backend.rag.engine` (avoiding a circular import) and stays trivially testable.

Logging failures here must never break the user-facing request — this is
observability, not a correctness dependency — so every failure mode is caught,
logged, and swallowed.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.audit import append_audit_log
from backend.core.logging import get_logger
from backend.models.query_log import RagQueryLog
from backend.security.pii import redact

logger = get_logger(__name__)

_MAX_QUESTION_LENGTH = 4000


class RagQueryAuditor:
    async def log(
        self,
        db: AsyncSession,
        *,
        user_id: str,
        conversation_id: str | None,
        question: str,
        retrieval_engine: str,
        num_context_chunks: int = 0,
        confidence_score: float = 0.0,
        confidence_level: str = "low",
        citations: list[dict[str, Any]] | None = None,
        documents_used: list[dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        stage_durations_ms: dict[str, float] | None = None,
        total_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        redacted_question = redact(question)[:_MAX_QUESTION_LENGTH]

        try:
            db.add(
                RagQueryLog(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    question=redacted_question,
                    retrieval_engine=retrieval_engine,
                    num_context_chunks=num_context_chunks,
                    confidence_score=confidence_score,
                    confidence_level=confidence_level,
                    citations_json=json.dumps(citations or []),
                    documents_used_json=json.dumps(documents_used or []),
                    warnings_json=json.dumps(warnings or []),
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    stage_durations_json=json.dumps(stage_durations_ms or {}),
                    total_ms=total_ms,
                    error=error[:4000] if error else None,
                )
            )
            await append_audit_log(
                db,
                user_id=user_id,
                action="rag_query_failed" if error else "rag_query",
                resource=conversation_id,
                detail=redacted_question,
            )
            await db.commit()
        except Exception:  # noqa: BLE001 - audit logging must never break the request
            logger.exception("rag_query_audit_log_failed")
            await db.rollback()
