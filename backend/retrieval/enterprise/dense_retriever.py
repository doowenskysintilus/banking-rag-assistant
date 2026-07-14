"""LangChain `BaseRetriever` wrapping our existing FAISS/Pinecone `VectorStoreBase`
Strategy — this is what lets `EnsembleRetriever`, `MultiQueryRetriever`, and
`ContextualCompressionRetriever` compose with our own dense search without any of
them knowing it isn't a native LangChain vector store.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

import numpy as np
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from backend.retrieval.enterprise.documents import hydrate_search_results
from backend.retrieval.vector_store.types import Filter, VectorStoreBase


@runtime_checkable
class QueryEmbedder(Protocol):
    """Structural type for anything that can embed a query — matched by
    `backend.retrieval.embeddings.service.EmbeddingService` and by lightweight
    test doubles alike, so this retriever doesn't hard-depend on the concrete
    multi-provider embedding service class.
    """

    async def embed_query(self, text: str) -> np.ndarray: ...


class DenseRetriever(BaseRetriever):
    """Embeds the query, searches the vector store, and hydrates results with
    real chunk content from the database (re-checking ownership there too — see
    `backend.retrieval.enterprise.documents.hydrate_search_results`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: VectorStoreBase
    embedding_service: QueryEmbedder
    db: Any  # AsyncSession — typed Any because BaseRetriever needs pydantic-friendly fields
    owner_id: str
    top_k: int = 20
    filters: Filter | None = None

    async def _retrieve_documents(self, query: str) -> list[Document]:
        query_vector = await self.embedding_service.embed_query(query)
        hits = await self.vector_store.search(
            query_vector, top_k=self.top_k, filters=self.filters
        )
        db: AsyncSession = self.db
        return await hydrate_search_results(db, hits, self.owner_id)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        return await self._retrieve_documents(query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._retrieve_documents(query))
        raise RuntimeError(
            "DenseRetriever can't run its sync entry point inside an already-running "
            "event loop (e.g. FastAPI) — use `.ainvoke()` / `aget_relevant_documents()`."
        )
