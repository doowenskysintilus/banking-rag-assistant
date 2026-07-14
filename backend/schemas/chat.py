from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from backend.models.conversation import MessageRole
from backend.schemas.common import ORMModel


class ChatQuery(BaseModel):
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation to continue, or omit to start a new one.",
    )
    question: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    """One reranked source chunk that backed the answer — `index` matches the
    "[Source N]" marker the LLM was instructed to reference in its response.
    """

    index: int
    document_id: str
    chunk_id: str
    document_filename: str
    excerpt: str
    score: float
    header_path: str = ""


class DocumentUsed(BaseModel):
    """Document-level rollup of the citations above — "which documents did this
    answer actually draw from", deduplicated, for a sources panel distinct from
    the more granular per-chunk citation list.
    """

    document_id: str
    filename: str
    chunk_count: int


class ChatAnswer(BaseModel):
    conversation_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    documents_used: list[DocumentUsed] = Field(default_factory=list)
    confidence_score: float = Field(
        description="0-1 retrieval-grounded confidence estimate — see backend.rag.confidence."
    )
    confidence_level: Literal["high", "medium", "low"]
    warnings: list[str] = Field(
        default_factory=list,
        description="Rule-based anti-hallucination flags from backend.rag.guardrails — empty when nothing looked off.",
    )
    model: str
    input_tokens: int
    output_tokens: int


class MessagePublic(ORMModel):
    id: str
    role: MessageRole
    content: str
    created_at: datetime


class ConversationPublic(ORMModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationPublic):
    messages: list[MessagePublic] = Field(default_factory=list)
