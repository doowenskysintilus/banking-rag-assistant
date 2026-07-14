"""EnterpriseRetrievalService — composes LangChain retrievers into a single
retrieval pipeline on top of our existing embeddings + vector store:

    dense retriever  ─┐
                       ├─ EnsembleRetriever (weighted RRF = hybrid search + score fusion)
    BM25 retriever   ─┘              │
                                     ▼ (optional)
                          MultiQueryRetriever (LLM-expanded query variants)
                                     │
                                     ▼ (optional)
              ContextualCompressionRetriever + reranking.RerankingService (BGE/cross-encoder)
                                     │
                                     ▼ (optional)
                              IntelligentRanker (business-rule boosts)
                                     │
                                     ▼
                          list[RetrievalResult], truncated to top_k

Every feature past the ensemble is individually toggleable via `Settings` — the
constructor reads those once; `retrieve()` is what runs per query.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.retrievers import BaseRetriever
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.retrieval.enterprise.bm25_corpus import get_owner_bm25_retriever
from backend.retrieval.enterprise.dense_retriever import DenseRetriever
from backend.retrieval.enterprise.intelligent_ranker import IntelligentRanker
from backend.retrieval.enterprise.reranker import LangChainRerankerCompressor
from backend.retrieval.embeddings.service import EmbeddingService
from backend.retrieval.reranking.service import (
    RerankingService,
    build_reranker_provider,
)
from backend.retrieval.vector_store.filters import matches
from backend.retrieval.vector_store.types import Filter, VectorStoreBase

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    document_id: str
    chunk_id: str
    content: str
    score: float
    header_path: str = ""


def _combine_owner_filter(owner_id: str, filters: Filter | None) -> Filter:
    owner_filter: Filter = {"owner_id": owner_id}
    if not filters:
        return owner_filter
    return {"$and": [owner_filter, filters]}


class EnterpriseRetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreBase,
        settings: Settings,
    ) -> None:
        self._embeddings = embedding_service
        self._vector_store = vector_store

        self._dense_top_k = settings.enterprise_dense_top_k
        self._bm25_top_k = settings.enterprise_bm25_top_k
        self._ensemble_weights = list(settings.enterprise_ensemble_weights)
        self._bm25_cache_ttl_seconds = settings.enterprise_bm25_cache_ttl_seconds

        self._multi_query_enabled = settings.enterprise_multi_query_enabled
        self._multi_query_count = settings.enterprise_multi_query_count
        self._chat_model = settings.openai_chat_model
        self._openai_api_key = settings.openai_api_key

        self._reranker: LangChainRerankerCompressor | None = None
        self._rerank_top_n = settings.enterprise_rerank_top_n
        if settings.enterprise_reranking_enabled:
            reranking_service = RerankingService(
                build_reranker_provider(settings),
                top_n=settings.enterprise_rerank_top_n,
                min_relevance_score=settings.enterprise_reranker_min_relevance_score,
            )
            self._reranker = LangChainRerankerCompressor(
                reranking_service=reranking_service
            )

        self._intelligent_ranker: IntelligentRanker | None = None
        if settings.enterprise_intelligent_ranking_enabled:
            self._intelligent_ranker = IntelligentRanker.with_default_rules(
                recency_weight=settings.enterprise_recency_boost_weight,
                recency_half_life_days=settings.enterprise_recency_half_life_days,
                structural_context_weight=settings.enterprise_structural_context_boost_weight,
            )

    async def _build_base_retriever(
        self, db: AsyncSession, owner_id: str, combined_filters: Filter
    ) -> BaseRetriever:
        dense_retriever = DenseRetriever(
            vector_store=self._vector_store,
            embedding_service=self._embeddings,
            db=db,
            owner_id=owner_id,
            top_k=self._dense_top_k,
            filters=combined_filters,
        )

        bm25_retriever = await get_owner_bm25_retriever(
            db,
            owner_id,
            top_k=self._bm25_top_k,
            ttl_seconds=self._bm25_cache_ttl_seconds,
        )

        if bm25_retriever is None:
            return dense_retriever

        from langchain_classic.retrievers.ensemble import EnsembleRetriever

        return EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=self._ensemble_weights,
            id_key="vector_id",
        )

    def _wrap_with_multi_query(self, base_retriever: BaseRetriever) -> BaseRetriever:
        if not self._multi_query_enabled:
            return base_retriever

        from langchain_classic.retrievers.multi_query import MultiQueryRetriever
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=self._chat_model,
            api_key=self._openai_api_key or None,
            temperature=0,
            # `langchain_openai.ChatOpenAI`'s own constructor param is still
            # named `max_tokens` (it handles translating to whatever the
            # underlying Chat Completions API call actually needs) — unlike
            # backend.llm.client.OpenAIClient's direct SDK call, which uses
            # the raw API's current `max_completion_tokens` field name.
            max_tokens=1024,
        )
        return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

    def _wrap_with_reranker(self, base_retriever: BaseRetriever) -> BaseRetriever:
        if self._reranker is None:
            return base_retriever

        from langchain_classic.retrievers.contextual_compression import (
            ContextualCompressionRetriever,
        )

        return ContextualCompressionRetriever(
            base_compressor=self._reranker, base_retriever=base_retriever
        )

    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        owner_id: str,
        filters: Filter | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        combined_filters = _combine_owner_filter(owner_id, filters)

        retriever = await self._build_base_retriever(db, owner_id, combined_filters)
        retriever = self._wrap_with_multi_query(retriever)
        retriever = self._wrap_with_reranker(retriever)

        documents = await retriever.ainvoke(query)

        # BM25Retriever has no native metadata filtering — its contribution to the
        # (already owner-scoped) ensemble still needs any *extra* filters applied.
        if filters:
            documents = [doc for doc in documents if matches(doc.metadata, filters)]

        if self._intelligent_ranker is not None and documents:
            documents = self._intelligent_ranker.rank(documents)

        effective_top_k = top_k or (
            self._rerank_top_n if self._reranker else self._dense_top_k
        )
        documents = documents[:effective_top_k]

        return [
            RetrievalResult(
                document_id=doc.metadata.get("document_id", ""),
                chunk_id=doc.metadata.get("chunk_id", ""),
                content=doc.page_content,
                score=float(
                    doc.metadata.get(
                        "final_score",
                        doc.metadata.get(
                            "rerank_score", doc.metadata.get("score", 0.0)
                        ),
                    )
                ),
                header_path=doc.metadata.get("header_path") or "",
            )
            for doc in documents
        ]
