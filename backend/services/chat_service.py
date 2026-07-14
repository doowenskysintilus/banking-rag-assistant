"""Conversation management and RAG query orchestration."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.core.exceptions import AppError, NotFoundError
from backend.core.logging import get_logger
from backend.llm.client import ChatTurn
from backend.models.conversation import Conversation, Message, MessageRole
from backend.monitoring.metrics import LLM_TOKENS_TOTAL
from backend.rag.context_builder import ContextSource
from backend.rag.engine import DocumentUsed as EngineDocumentUsed
from backend.rag.engine import PreparedQuery, RAGEngine
from backend.schemas.chat import ChatAnswer, Citation, DocumentUsed

logger = get_logger(__name__)

MAX_HISTORY_TURNS = 10


class ChatService:
    """Handles both RAG queries and conversation history.

    `rag_engine` is only required by `ask`/`ask_stream`. Conversation listing/reading
    is pure DB access, so callers that only need those methods (e.g. the conversation
    history endpoints) can construct this without paying the cost of initializing the
    embedding model and vector store — see `get_conversation_service` in
    `backend.core.dependencies`.
    """

    def __init__(self, db: AsyncSession, rag_engine: RAGEngine | None = None) -> None:
        self._db = db
        self._rag = rag_engine

    def _require_rag_engine(self) -> RAGEngine:
        if self._rag is None:
            raise RuntimeError(
                "ChatService was constructed without a RAGEngine; cannot answer queries."
            )
        return self._rag

    async def _get_or_create_conversation(
        self, user_id: str, conversation_id: str | None
    ) -> Conversation:
        if conversation_id is None:
            conversation = Conversation(user_id=user_id)
            self._db.add(conversation)
            await self._db.flush()
            return conversation

        conversation = await self._db.get(Conversation, conversation_id)
        if conversation is None or conversation.user_id != user_id:
            raise NotFoundError(f"Conversation {conversation_id} was not found")
        return conversation

    async def _load_history(self, conversation_id: str) -> list[ChatTurn]:
        result = await self._db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(MAX_HISTORY_TURNS)
        )
        messages = list(reversed(result.scalars().all()))
        return [ChatTurn(role=m.role.value, content=m.content) for m in messages]

    @staticmethod
    def _to_citations(sources: list[ContextSource]) -> list[Citation]:
        return [
            Citation(
                index=source.index,
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                document_filename=source.document_filename,
                excerpt=source.excerpt,
                score=round(source.score, 4),
                header_path=source.header_path,
            )
            for source in sources
        ]

    @staticmethod
    def _to_documents_used(
        documents_used: list[EngineDocumentUsed],
    ) -> list[DocumentUsed]:
        return [
            DocumentUsed(
                document_id=doc.document_id,
                filename=doc.filename,
                chunk_count=doc.chunk_count,
            )
            for doc in documents_used
        ]

    async def ask(
        self, user_id: str, conversation_id: str | None, question: str
    ) -> ChatAnswer:
        conversation = await self._get_or_create_conversation(user_id, conversation_id)
        history = await self._load_history(conversation.id) if conversation_id else []

        self._db.add(
            Message(
                conversation_id=conversation.id, role=MessageRole.USER, content=question
            )
        )

        response = await self._require_rag_engine().answer(
            self._db,
            question,
            user_id,
            conversation_id=conversation.id,
            history=history,
        )

        citations = self._to_citations(response.sources)
        documents_used = self._to_documents_used(response.documents_used)

        self._db.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=response.answer,
                citations=json.dumps([c.model_dump() for c in citations]),
            )
        )
        await self._db.commit()

        LLM_TOKENS_TOTAL.labels("input").inc(response.input_tokens)
        LLM_TOKENS_TOTAL.labels("output").inc(response.output_tokens)

        return ChatAnswer(
            conversation_id=conversation.id,
            answer=response.answer,
            citations=citations,
            documents_used=documents_used,
            confidence_score=response.confidence.score,
            confidence_level=response.confidence.level,
            warnings=response.warnings,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    async def ask_stream(
        self, user_id: str, conversation_id: str | None, question: str
    ) -> AsyncIterator[str]:
        """Yields SSE-formatted text chunks, then persists the final message once complete."""
        conversation = await self._get_or_create_conversation(user_id, conversation_id)
        history = await self._load_history(conversation.id) if conversation_id else []

        self._db.add(
            Message(
                conversation_id=conversation.id, role=MessageRole.USER, content=question
            )
        )
        await self._db.flush()

        yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation.id})}\n\n"

        engine = self._require_rag_engine()
        t_start = time.perf_counter()
        prepared: PreparedQuery | None = None

        # Once headers are sent, FastAPI's exception handlers can no longer turn a
        # mid-stream error into a proper HTTP error response — so this loop must
        # catch failures itself and emit an SSE "error" event instead of raising.
        try:
            prepared, stream = await engine.answer_stream(
                self._db, question, user_id, history=history
            )

            async for chunk in stream:
                if chunk.is_final and chunk.final_response is not None:
                    llm_response = chunk.final_response
                    citations = self._to_citations(prepared.sources)
                    documents_used = self._to_documents_used(prepared.documents_used)
                    warnings = engine.compute_warnings(prepared, llm_response)

                    self._db.add(
                        Message(
                            conversation_id=conversation.id,
                            role=MessageRole.ASSISTANT,
                            content=llm_response.text,
                            citations=json.dumps([c.model_dump() for c in citations]),
                        )
                    )
                    await self._db.commit()

                    LLM_TOKENS_TOTAL.labels("input").inc(llm_response.input_tokens)
                    LLM_TOKENS_TOTAL.labels("output").inc(llm_response.output_tokens)

                    await engine.log_query(
                        self._db,
                        user_id=user_id,
                        conversation_id=conversation.id,
                        question=question,
                        prepared=prepared,
                        llm_response=llm_response,
                        warnings=warnings,
                        total_ms=(time.perf_counter() - t_start) * 1000,
                    )

                    payload = {
                        "citations": [c.model_dump() for c in citations],
                        "documents_used": [d.model_dump() for d in documents_used],
                        "confidence_score": prepared.confidence.score,
                        "confidence_level": prepared.confidence.level,
                        "warnings": warnings,
                        "model": llm_response.model,
                        "input_tokens": llm_response.input_tokens,
                        "output_tokens": llm_response.output_tokens,
                    }
                    yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                else:
                    yield f"event: delta\ndata: {json.dumps({'text': chunk.text})}\n\n"
        except AppError as exc:
            await self._db.rollback()
            logger.warning(
                "chat_stream_app_error", extra={"error_code": exc.error_code}
            )
            await engine.log_query(
                self._db,
                user_id=user_id,
                conversation_id=conversation.id,
                question=question,
                prepared=prepared,
                error=exc.message,
                total_ms=(time.perf_counter() - t_start) * 1000,
            )
            yield f"event: error\ndata: {json.dumps({'code': exc.error_code, 'message': exc.message})}\n\n"
        except Exception as exc:  # noqa: BLE001 - never let a raw exception escape the SSE body
            await self._db.rollback()
            logger.exception("chat_stream_unexpected_error")
            await engine.log_query(
                self._db,
                user_id=user_id,
                conversation_id=conversation.id,
                question=question,
                prepared=prepared,
                error=str(exc),
                total_ms=(time.perf_counter() - t_start) * 1000,
            )
            yield (
                "event: error\ndata: "
                + json.dumps(
                    {
                        "code": "internal_error",
                        "message": "An unexpected error occurred.",
                    }
                )
                + "\n\n"
            )

    async def list_conversations(self, user_id: str) -> list[Conversation]:
        result = await self._db.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
        )
        return list(result.scalars().all())

    async def get_conversation(
        self, user_id: str, conversation_id: str
    ) -> Conversation:
        result = await self._db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .options(selectinload(Conversation.messages))
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            raise NotFoundError(f"Conversation {conversation_id} was not found")
        return conversation
