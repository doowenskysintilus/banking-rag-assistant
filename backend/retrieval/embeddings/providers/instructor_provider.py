"""Instructor-XL provider — an instruction-tuned embedding model (hkunlp/instructor-xl)
run via the `InstructorEmbedding` package.

Unlike BGE (which only prefixes queries), Instructor-XL embeds explicit
(instruction, text) pairs on *both* sides of retrieval, with a different
instruction for documents vs. queries — that asymmetry is the whole point of the
model, so both instructions are configurable (see Settings).
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.retrieval.embeddings.types import EmbeddingProvider, l2_normalize

logger = get_logger(__name__)


class InstructorXLProvider(EmbeddingProvider):
    name = "instructor_xl"

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.instructor_model_name
        self.dimension = settings.instructor_embedding_dim
        self._document_instruction = settings.instructor_document_instruction
        self._query_instruction = settings.instructor_query_instruction
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                from InstructorEmbedding import INSTRUCTOR

                logger.info(
                    "loading_embedding_model",
                    extra={"provider": self.name, "model": self._model_name},
                )
                self._model = INSTRUCTOR(self._model_name)

    def _encode_sync(self, texts: list[str], is_query: bool) -> np.ndarray:
        self._ensure_loaded()
        instruction = (
            self._query_instruction if is_query else self._document_instruction
        )
        pairs = [[instruction, text] for text in texts]
        vectors = np.asarray(self._model.encode(pairs), dtype="float32")
        return l2_normalize(vectors)

    async def embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        return await asyncio.to_thread(self._encode_sync, texts, is_query)

    @property
    def is_available(self) -> bool:
        try:
            import InstructorEmbedding  # noqa: F401
        except ImportError:
            return False
        return True
