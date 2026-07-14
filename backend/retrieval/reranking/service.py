"""RerankingService — the entry point the rest of the app depends on.

Beyond just reordering candidates by relevance, this is where "reduce
hallucinations" becomes concrete: any candidate whose calibrated relevance score
falls below `min_relevance_score` is dropped entirely, not just ranked lower. If
*nothing* clears the bar, `RerankResult.low_confidence` comes back `True` and
`documents` is empty — callers (see `backend.retrieval.enterprise.reranker`) then
pass no context to the LLM at all, which (per `backend.llm.prompts.SYSTEM_PROMPT`)
makes it say "I don't have enough information" instead of guessing from noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.core.config import Settings, get_settings
from backend.core.exceptions import ExternalServiceError
from backend.retrieval.reranking.bge_reranker import BGERerankerProvider
from backend.retrieval.reranking.cross_encoder import CrossEncoderRerankerProvider
from backend.retrieval.reranking.selector import (
    probe_availability,
    select_provider_name,
)
from backend.retrieval.reranking.types import RerankerProvider

_PROVIDER_CLASSES: dict[str, type[RerankerProvider]] = {
    "bge": BGERerankerProvider,
    "cross_encoder": CrossEncoderRerankerProvider,
}


def build_reranker_provider(settings: Settings) -> RerankerProvider:
    availability = probe_availability(settings)
    name = select_provider_name(settings, availability)
    provider_cls = _PROVIDER_CLASSES.get(name)
    if provider_cls is None:
        raise ExternalServiceError(f"Unknown reranker provider: {name}")
    return provider_cls(settings)


@dataclass
class RerankedDocument:
    content: str
    original_index: int
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    documents: list[RerankedDocument]
    low_confidence: bool
    provider_name: str


class RerankingService:
    def __init__(
        self,
        provider: RerankerProvider,
        *,
        top_n: int = 5,
        min_relevance_score: float = 0.0,
    ) -> None:
        self._provider = provider
        self._top_n = top_n
        self._min_relevance_score = min_relevance_score

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def rerank(
        self,
        query: str,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> RerankResult:
        if not documents:
            return RerankResult(
                documents=[], low_confidence=False, provider_name=self._provider.name
            )

        scores = await self._provider.score(query, documents)

        candidates = [
            RerankedDocument(
                content=doc,
                original_index=i,
                relevance_score=float(score),
                metadata=dict(metadatas[i]) if metadatas else {},
            )
            for i, (doc, score) in enumerate(zip(documents, scores, strict=True))
        ]
        candidates.sort(key=lambda c: c.relevance_score, reverse=True)

        confident = [
            c for c in candidates if c.relevance_score >= self._min_relevance_score
        ]
        low_confidence = not confident

        return RerankResult(
            documents=confident[: self._top_n],
            low_confidence=low_confidence,
            provider_name=self._provider.name,
        )


_service_singleton: RerankingService | None = None


def get_reranking_service() -> RerankingService:
    global _service_singleton
    if _service_singleton is None:
        settings = get_settings()
        provider = build_reranker_provider(settings)
        _service_singleton = RerankingService(
            provider,
            top_n=settings.enterprise_rerank_top_n,
            min_relevance_score=settings.enterprise_reranker_min_relevance_score,
        )
    return _service_singleton
