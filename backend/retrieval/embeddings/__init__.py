"""Multi-provider embedding generation: OpenAI / BGE-Large / Instructor-XL, with
automatic provider selection, batching, bounded parallelism, and a Redis-backed
cache. See `backend.retrieval.embeddings.service.EmbeddingService` for the
orchestrator and `backend.retrieval.embeddings.selector` for the auto-selection
policy.
"""

from backend.retrieval.embeddings.service import (
    EmbeddingService,
    build_cache,
    build_provider,
    get_embedding_service,
)
from backend.retrieval.embeddings.types import EmbeddingProvider

__all__ = [
    "EmbeddingService",
    "EmbeddingProvider",
    "build_cache",
    "build_provider",
    "get_embedding_service",
]
