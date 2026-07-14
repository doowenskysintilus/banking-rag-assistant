"""BGE-Large provider — a strong open-source local model (BAAI/bge-large-en-v1.5),
run via sentence-transformers.

BGE embeddings are asymmetric: the official model card recommends prefixing
*queries* with an instruction so the retrieval task is encoded into the embedding,
while *documents* are embedded as-is.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.retrieval.embeddings.types import EmbeddingProvider

logger = get_logger(__name__)

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGELargeProvider(EmbeddingProvider):
    name = "bge_large"

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.bge_model_name
        self.dimension = settings.bge_embedding_dim
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "loading_embedding_model",
                    extra={"provider": self.name, "model": self._model_name},
                )
                self._model = SentenceTransformer(self._model_name)

    def _encode_sync(self, texts: list[str], is_query: bool) -> np.ndarray:
        self._ensure_loaded()
        inputs = (
            [f"{_QUERY_INSTRUCTION}{text}" for text in texts] if is_query else texts
        )
        vectors = self._model.encode(
            inputs, normalize_embeddings=True, convert_to_numpy=True
        )
        return vectors.astype("float32")

    async def embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        return await asyncio.to_thread(self._encode_sync, texts, is_query)

    @property
    def is_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True
