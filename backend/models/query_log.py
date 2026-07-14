"""Persisted audit trail: one row per RAG query, success or failure.

Separate from `Message`/`Conversation` (the user-facing chat history) — this is a
compliance-oriented record that must survive independently of whether the
conversational bookkeeping succeeds, and captures retrieval/ranking internals
(scores, timings, engine used) that never need to reach the chat UI.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class RagQueryLog(Base):
    __tablename__ = "rag_query_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    conversation_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_engine: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "native" | "enterprise"

    num_context_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence_level: Mapped[str] = mapped_column(
        String(10), nullable=False, default="low"
    )

    # JSON-encoded lists — see backend.rag.context_builder.ContextSource /
    # backend.rag.engine.DocumentUsed for the shapes serialized here.
    citations_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    documents_used_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    # JSON-encoded list of strings from backend.rag.guardrails.verify_answer —
    # rule-based anti-hallucination flags (fabricated citation, unverified numeric
    # claim, unwarranted confidence, possible legal overreach). Empty when clean.
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    model: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # JSON-encoded {"retrieval_ms": ..., "reranking_ms": ..., "context_builder_ms":
    # ..., "prompt_builder_ms": ..., "llm_ms": ...} — whichever stages actually ran.
    stage_durations_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )
    total_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Populated only if the query failed partway through — question/timings up to
    # the failure point are still logged (see backend.rag.audit.RagQueryAuditor).
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
