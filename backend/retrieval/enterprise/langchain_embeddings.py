"""Adapts our multi-provider `EmbeddingService` to LangChain's `Embeddings`
interface, so LangChain retrievers/tools can use it without knowing anything
about OpenAI/BGE/Instructor-XL or the Redis cache underneath.

`Embeddings` is a plain ABC (not a pydantic model), so this is a thin, direct
wrapper — no field-declaration ceremony needed.
"""

from __future__ import annotations

import asyncio

from langchain_core.embeddings import Embeddings

from backend.retrieval.embeddings.service import EmbeddingService


class LangChainEmbeddingsAdapter(Embeddings):
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embedding_service = embedding_service

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = await self._embedding_service.embed_texts(texts)
        return vectors.tolist()

    async def aembed_query(self, text: str) -> list[float]:
        vector = await self._embedding_service.embed_query(text)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run_sync(lambda: self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._run_sync(lambda: self.aembed_query(text))

    @staticmethod
    def _run_sync(make_coro):
        """Only safe to call outside a running event loop (e.g. a standalone
        script). Inside FastAPI's async request handlers, an already-running
        loop means `asyncio.run()` would deadlock rather than help — callers
        there must use `aembed_documents`/`aembed_query` directly.

        Takes a zero-arg callable that *creates* the coroutine (rather than an
        already-constructed one) so the raise-inside-a-running-loop branch below
        never leaves a never-awaited coroutine behind.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(make_coro())
        raise RuntimeError(
            "LangChainEmbeddingsAdapter's sync embed_documents/embed_query can't run "
            "inside an already-running event loop. Use aembed_documents/aembed_query instead."
        )
