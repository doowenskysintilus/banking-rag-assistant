"""LangChain-facing reranking compressor.

Delegates to `backend.retrieval.reranking.RerankingService` (BGE / generic
cross-encoder, auto-selected) rather than loading a model directly — the same
benchmarked/evaluated reranking logic, including the min-relevance-score
hallucination guard, runs whether it's invoked here (inside a LangChain
`ContextualCompressionRetriever`) or used standalone.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict

from backend.core.logging import get_logger
from backend.retrieval.reranking.service import RerankingService

logger = get_logger(__name__)


class LangChainRerankerCompressor(BaseDocumentCompressor):
    """Drop into `ContextualCompressionRetriever(base_compressor=..., base_retriever=...)`
    and every wrapped retriever's results get reranked (and confidence-filtered)
    transparently.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reranking_service: RerankingService

    async def acompress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Any = None
    ) -> Sequence[Document]:
        return await self._compress(documents, query)

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Any = None
    ) -> Sequence[Document]:
        # `BaseDocumentCompressor.acompress_documents` defaults to running this
        # sync method inside a thread-pool executor when called from async code —
        # a fresh thread has no event loop of its own, so `asyncio.run()` here is
        # safe in both the direct-sync-call case and that executor case.
        return asyncio.run(self._compress(documents, query))

    async def _compress(
        self, documents: Sequence[Document], query: str
    ) -> Sequence[Document]:
        if not documents:
            return []

        documents_list = list(documents)
        result = await self.reranking_service.rerank(
            query,
            [doc.page_content for doc in documents_list],
            metadatas=[doc.metadata for doc in documents_list],
        )

        if result.low_confidence:
            logger.warning(
                "reranker_low_confidence_no_context",
                extra={
                    "provider": self.reranking_service.provider_name,
                    "candidates": len(documents_list),
                },
            )

        return [
            Document(
                page_content=item.content,
                metadata={**item.metadata, "rerank_score": item.relevance_score},
            )
            for item in result.documents
        ]
