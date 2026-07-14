"""The main RAG engine — the explicit pipeline every question goes through:

    Question utilisateur
        → Retrieval            (backend.retrieval.vector_store / .enterprise)
        → Reranking             (backend.retrieval.reranking)
        → Context Builder         (backend.rag.context_builder)
        → Prompt Builder            (backend.rag.prompt_builder)
        → LLM                         (backend.llm.client.OpenAIClient)
        → Réponse                       (RAGResponse: answer + citations + confidence)

Every stage is timed; every query — success or failure — is persisted via
`backend.rag.audit.RagQueryAuditor`.

Retrieval + Reranking are two separate, separately-timed calls in "native" mode
(`settings.rag_retrieval_engine == "native"`: our own FAISS/Pinecone hybrid search,
then `RerankingService`). In "enterprise" mode, `EnterpriseRetrievalService`
already performs LangChain-based hybrid retrieval *and* reranking internally (plus
optional multi-query expansion) as one call — that's reflected honestly here as a
single combined stage rather than faking a split that didn't happen.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.llm.client import ChatTurn, LLMResponse, LLMStreamChunk, OpenAIClient
from backend.models.document import Document, DocumentChunk
from backend.rag.audit import RagQueryAuditor
from backend.rag.confidence import ConfidenceAssessment, assess_confidence
from backend.rag.context_builder import ContextBuilder, ContextSource, SourceChunk
from backend.rag.guardrails import verify_answer
from backend.rag.prompt_builder import PromptBuilder
from backend.retrieval.embeddings import EmbeddingService
from backend.retrieval.reranking.service import RerankingService
from backend.retrieval.vector_store import VectorStoreBase

if TYPE_CHECKING:
    # Only imported for type checking — the "native" engine has no LangChain
    # dependency at runtime. See backend.core.dependencies for the (lazy) real
    # import that constructs one when rag_retrieval_engine="enterprise".
    from backend.retrieval.enterprise import EnterpriseRetrievalService

logger = get_logger(__name__)

_OVER_FETCH_FACTOR = 4  # native mode: candidates fetched before reranking cuts to top_k


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


@dataclass
class RetrievedCandidate:
    document_id: str
    chunk_id: str
    content: str
    score: float
    header_path: str = ""


@dataclass
class DocumentUsed:
    document_id: str
    filename: str
    chunk_count: int


@dataclass
class PreparedQuery:
    """Output of stages 1-4 — everything needed to call the LLM and, once it
    responds, assemble the final `RAGResponse`.
    """

    messages: list[dict]
    context_text: str
    source_texts: list[str]
    sources: list[ContextSource]
    documents_used: list[DocumentUsed]
    confidence: ConfidenceAssessment
    stage_durations_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class RAGResponse:
    answer: str
    sources: list[ContextSource]
    source_texts: list[str]
    documents_used: list[DocumentUsed]
    confidence: ConfidenceAssessment
    warnings: list[str]
    model: str
    input_tokens: int
    output_tokens: int
    stage_durations_ms: dict[str, float]


class RAGEngine:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreBase,
        reranking_service: RerankingService,
        openai_client: OpenAIClient,
        settings: Settings,
        enterprise_retrieval: "EnterpriseRetrievalService | None" = None,
    ) -> None:
        self._embeddings = embedding_service
        self._vector_store = vector_store
        self._reranking = reranking_service
        self._llm = openai_client
        self._context_builder = ContextBuilder()
        self._prompt_builder = PromptBuilder()
        self._auditor = RagQueryAuditor()

        self._retrieval_engine = settings.rag_retrieval_engine
        self._retrieval_mode = settings.rag_retrieval_mode
        self._top_k = settings.rag_top_k
        self._candidate_k = max(self._top_k * _OVER_FETCH_FACTOR, self._top_k)
        self._enterprise_retrieval = enterprise_retrieval

        if (
            self._retrieval_engine == "enterprise"
            and self._enterprise_retrieval is None
        ):
            raise ValueError(
                "rag_retrieval_engine='enterprise' requires an EnterpriseRetrievalService instance "
                "(see backend.core.dependencies.get_rag_engine)."
            )

    # --- Stage 1: Retrieval --------------------------------------------------

    async def _retrieve_candidates(
        self, db: AsyncSession, question: str, owner_id: str
    ) -> tuple[list[RetrievedCandidate], bool]:
        """Returns (candidates, already_reranked)."""
        if self._retrieval_engine == "enterprise":
            results = await self._enterprise_retrieval.retrieve(
                db, question, owner_id, top_k=self._top_k
            )
            candidates = [
                RetrievedCandidate(
                    document_id=r.document_id,
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=r.score,
                    header_path=r.header_path,
                )
                for r in results
            ]
            return candidates, True

        query_vector = await self._embeddings.embed_query(question)
        # Every user only ever retrieves their own documents — this filter is what
        # keeps one customer's banking documents out of another customer's answers.
        filters = {"owner_id": owner_id}
        if self._retrieval_mode == "hybrid":
            hits = await self._vector_store.hybrid_search(
                query_vector, question, top_k=self._candidate_k, filters=filters
            )
        else:
            hits = await self._vector_store.search(
                query_vector, top_k=self._candidate_k, filters=filters
            )

        if not hits:
            return [], False

        # Re-asserted here via a SQL join against the authoritative Document row —
        # defense in depth in case the vector store's metadata sidecar (a JSON
        # file) ever drifted from the database, the source of truth for ownership.
        vector_ids = [hit.vector_id for hit in hits]
        stmt = (
            select(DocumentChunk)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.vector_id.in_(vector_ids), Document.owner_id == owner_id
            )
        )
        result = await db.execute(stmt)
        chunks_by_vector_id = {
            chunk.vector_id: chunk for chunk in result.scalars().all()
        }

        candidates = []
        for hit in hits:
            chunk = chunks_by_vector_id.get(hit.vector_id)
            if chunk is None:
                # Stale vector id (document deleted) or an owner mismatch caught
                # by the defense-in-depth check above — skip rather than leak.
                continue
            candidates.append(
                RetrievedCandidate(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    content=chunk.content,
                    score=hit.score,
                    header_path=chunk.header_path,
                )
            )
        return candidates, False

    # --- Stage 2: Reranking ---------------------------------------------------

    async def _rerank(
        self, question: str, candidates: list[RetrievedCandidate]
    ) -> tuple[list[RetrievedCandidate], bool]:
        if not candidates:
            return [], False

        result = await self._reranking.rerank(
            question,
            [c.content for c in candidates],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "header_path": c.header_path,
                }
                for c in candidates
            ],
        )
        reranked = [
            RetrievedCandidate(
                document_id=item.metadata["document_id"],
                chunk_id=item.metadata["chunk_id"],
                content=item.content,
                score=item.relevance_score,
                header_path=item.metadata.get("header_path", ""),
            )
            for item in result.documents
        ]
        return reranked, result.low_confidence

    # --- helpers shared by stage 3 -------------------------------------------

    async def _fetch_filenames(
        self, db: AsyncSession, chunks: list[RetrievedCandidate]
    ) -> dict[str, str]:
        document_ids = list({c.document_id for c in chunks})
        if not document_ids:
            return {}
        result = await db.execute(
            select(Document.id, Document.filename).where(Document.id.in_(document_ids))
        )
        return dict(result.all())

    @staticmethod
    def _documents_used(
        chunks: list[RetrievedCandidate], filenames: dict[str, str]
    ) -> list[DocumentUsed]:
        counts: dict[str, int] = {}
        order: list[str] = []
        for chunk in chunks:
            if chunk.document_id not in counts:
                order.append(chunk.document_id)
            counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
        return [
            DocumentUsed(
                document_id=doc_id,
                filename=filenames.get(doc_id, "Unknown document"),
                chunk_count=counts[doc_id],
            )
            for doc_id in order
        ]

    # --- Stages 1-4, run together as "prepare" -------------------------------

    async def prepare(
        self,
        db: AsyncSession,
        question: str,
        owner_id: str,
        history: list[ChatTurn] | None = None,
    ) -> PreparedQuery:
        durations: dict[str, float] = {}

        t0 = time.perf_counter()
        candidates, already_reranked = await self._retrieve_candidates(
            db, question, owner_id
        )
        durations["retrieval_ms"] = _elapsed_ms(t0)

        if already_reranked:
            # Enterprise mode's own reranker already applied its confidence
            # threshold; an empty result is the only "low confidence" signal we
            # can observe from here without reaching into that engine's internals.
            reranked, low_confidence = candidates, not candidates
        else:
            t1 = time.perf_counter()
            reranked, low_confidence = await self._rerank(question, candidates)
            durations["reranking_ms"] = _elapsed_ms(t1)

        t2 = time.perf_counter()
        filenames = await self._fetch_filenames(db, reranked)
        source_chunks = [
            SourceChunk(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                content=c.content,
                score=c.score,
                header_path=c.header_path,
            )
            for c in reranked
        ]
        context_result = self._context_builder.build(source_chunks, filenames)
        durations["context_builder_ms"] = _elapsed_ms(t2)

        # Computed here (not after prompt building) so the confidence *level* can
        # itself steer the prompt — a low-confidence retrieval gets an explicit
        # extra caution addendum (see backend.llm.prompts.templates).
        confidence = assess_confidence(
            [s.score for s in context_result.sources], low_confidence
        )
        documents_used = self._documents_used(reranked, filenames)

        t3 = time.perf_counter()
        messages = self._prompt_builder.build(
            question,
            context_result.context_text,
            history,
            confidence_level=confidence.level,
        )
        durations["prompt_builder_ms"] = _elapsed_ms(t3)

        return PreparedQuery(
            messages=messages,
            context_text=context_result.context_text,
            source_texts=[c.content for c in source_chunks],
            sources=context_result.sources,
            documents_used=documents_used,
            confidence=confidence,
            stage_durations_ms=durations,
        )

    # --- Stages 5-6 + audit logging ------------------------------------------

    @staticmethod
    def compute_warnings(
        prepared: PreparedQuery, llm_response: LLMResponse
    ) -> list[str]:
        """Post-hoc anti-hallucination checks (see backend.rag.guardrails) run
        against the model's *actual* answer — separate from `build_response` so
        the streaming path (which builds its own response shape) can call it too.
        """
        return verify_answer(
            llm_response.text,
            prepared.context_text,
            prepared.sources,
            prepared.confidence,
        )

    @classmethod
    def build_response(
        cls, prepared: PreparedQuery, llm_response: LLMResponse
    ) -> RAGResponse:
        return RAGResponse(
            answer=llm_response.text,
            sources=prepared.sources,
            source_texts=prepared.source_texts,
            documents_used=prepared.documents_used,
            confidence=prepared.confidence,
            warnings=cls.compute_warnings(prepared, llm_response),
            model=llm_response.model,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            stage_durations_ms=prepared.stage_durations_ms,
        )

    async def log_query(
        self,
        db: AsyncSession,
        *,
        user_id: str,
        conversation_id: str | None,
        question: str,
        prepared: PreparedQuery | None = None,
        llm_response: LLMResponse | None = None,
        warnings: list[str] | None = None,
        error: str | None = None,
        total_ms: float = 0.0,
    ) -> None:
        await self._auditor.log(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
            retrieval_engine=self._retrieval_engine,
            num_context_chunks=len(prepared.sources) if prepared else 0,
            confidence_score=prepared.confidence.score if prepared else 0.0,
            confidence_level=prepared.confidence.level if prepared else "low",
            citations=[asdict(s) for s in prepared.sources] if prepared else [],
            documents_used=[asdict(d) for d in prepared.documents_used]
            if prepared
            else [],
            warnings=warnings or [],
            model=llm_response.model if llm_response else "",
            input_tokens=llm_response.input_tokens if llm_response else 0,
            output_tokens=llm_response.output_tokens if llm_response else 0,
            stage_durations_ms=prepared.stage_durations_ms if prepared else {},
            total_ms=total_ms,
            error=error,
        )

    async def answer(
        self,
        db: AsyncSession,
        question: str,
        owner_id: str,
        *,
        conversation_id: str | None = None,
        history: list[ChatTurn] | None = None,
    ) -> RAGResponse:
        """Full six-stage pipeline, non-streaming. Always logs the query — on
        success or failure — before returning/raising.
        """
        t_start = time.perf_counter()
        prepared: PreparedQuery | None = None
        llm_response: LLMResponse | None = None
        response: RAGResponse | None = None
        error: str | None = None
        try:
            prepared = await self.prepare(db, question, owner_id, history)

            t_llm = time.perf_counter()
            llm_response = await self._llm.answer(prepared.messages)
            prepared.stage_durations_ms["llm_ms"] = _elapsed_ms(t_llm)

            response = self.build_response(prepared, llm_response)
            return response
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            await self.log_query(
                db,
                user_id=owner_id,
                conversation_id=conversation_id,
                question=question,
                prepared=prepared,
                llm_response=llm_response,
                warnings=response.warnings if response else [],
                error=error,
                total_ms=_elapsed_ms(t_start),
            )

    async def answer_stream(
        self,
        db: AsyncSession,
        question: str,
        owner_id: str,
        history: list[ChatTurn] | None = None,
    ) -> tuple[PreparedQuery, AsyncIterator[LLMStreamChunk]]:
        """Runs stages 1-4 (Retrieval → Reranking → Context Builder → Prompt
        Builder) and returns them alongside the streaming LLM generator (stage
        5). The caller (see `backend.services.chat_service.ChatService.ask_stream`)
        drives the stream and must call `log_query` itself once it completes (or
        fails) — only it knows when that actually happens.
        """
        prepared = await self.prepare(db, question, owner_id, history)
        stream = self._llm.answer_stream(prepared.messages)
        return prepared, stream

    @staticmethod
    async def document_filename(db: AsyncSession, document_id: str) -> str | None:
        result = await db.execute(
            select(Document.filename).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()
