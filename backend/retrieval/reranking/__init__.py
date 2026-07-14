"""Advanced reranking: BGE reranker + generic cross-encoder providers behind a
common `RerankerProvider` interface, with automatic provider selection, a
confidence threshold that drops low-relevance candidates entirely (the concrete
"reduce hallucinations" mechanism — see `backend.retrieval.reranking.service`),
and a metrics/evaluation/benchmark suite to measure the effect.
"""

from backend.retrieval.reranking.service import (
    RerankedDocument,
    RerankingService,
    RerankResult,
    build_reranker_provider,
    get_reranking_service,
)
from backend.retrieval.reranking.types import RerankerProvider

__all__ = [
    "RerankerProvider",
    "RerankedDocument",
    "RerankResult",
    "RerankingService",
    "build_reranker_provider",
    "get_reranking_service",
]
