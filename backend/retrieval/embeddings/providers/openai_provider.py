"""OpenAI Embeddings provider — a managed API, no local model to load.

Embeddings are symmetric (queries and documents use the same representation), so
`is_query` is accepted for interface compatibility but has no effect here.
"""

from __future__ import annotations

import numpy as np

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError, RateLimitedError
from backend.core.logging import get_logger
from backend.retrieval.embeddings.types import EmbeddingProvider, l2_normalize

logger = get_logger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.openai_api_key
        self._model = settings.openai_embedding_model
        self.dimension = settings.openai_embedding_dim
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key or None)
        return self._client

    async def embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        client = self._ensure_client()
        try:
            response = await client.embeddings.create(
                model=self._model, input=texts, dimensions=self.dimension
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as typed error below
            raise self._map_error(exc) from exc

        vectors = np.array([item.embedding for item in response.data], dtype="float32")
        return l2_normalize(vectors)

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    @staticmethod
    def _map_error(exc: Exception) -> Exception:
        import openai

        if isinstance(exc, openai.RateLimitError):
            logger.warning("openai_embeddings_rate_limited")
            return RateLimitedError(
                "The embedding service is temporarily rate limited. Please retry shortly."
            )
        if isinstance(exc, openai.AuthenticationError):
            logger.error("openai_embeddings_authentication_error")
            return ExternalServiceError("OpenAI embeddings authentication failed.")
        if isinstance(exc, openai.APIStatusError):
            logger.error(
                "openai_embeddings_api_status_error",
                extra={"status_code": exc.status_code},
            )
            if exc.status_code >= 500:
                return ExternalServiceError(
                    "The embedding service is temporarily unavailable."
                )
            return ExternalServiceError(f"Embedding request failed: {exc.message}")
        if isinstance(exc, openai.APIConnectionError):
            logger.error("openai_embeddings_connection_error")
            return ExternalServiceError(
                "Could not reach the OpenAI embeddings service."
            )
        logger.exception("openai_embeddings_unexpected_error")
        return ExternalServiceError(
            "Unexpected error calling the OpenAI embeddings service."
        )
