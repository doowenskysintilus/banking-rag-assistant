"""Generic cross-encoder reranker provider (e.g. MS-MARCO-trained models), via
`sentence_transformers.CrossEncoder`. Not BGE-specific — the manual-override
option for when a smaller/faster/differently-licensed model is preferred over
the BGE default.
"""

from __future__ import annotations

import asyncio
import threading

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.retrieval.reranking.types import RerankerProvider, sigmoid

logger = get_logger(__name__)


class CrossEncoderRerankerProvider(RerankerProvider):
    name = "cross_encoder"

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.cross_encoder_reranker_model
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                logger.info(
                    "loading_reranker_model",
                    extra={"provider": self.name, "model": self._model_name},
                )
                self._model = CrossEncoder(self._model_name)

    def _score_sync(self, query: str, documents: list[str]) -> list[float]:
        self._ensure_loaded()
        pairs = [[query, doc] for doc in documents]
        raw_scores = self._model.predict(pairs)
        # MS-MARCO-style cross-encoders output unbounded logits, not [0, 1]
        # probabilities — sigmoid puts them on the same scale BGE's normalize=True
        # uses, so a single min_relevance_score threshold works across providers.
        return [sigmoid(float(s)) for s in raw_scores]

    async def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return await asyncio.to_thread(self._score_sync, query, documents)

    @property
    def is_available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True
