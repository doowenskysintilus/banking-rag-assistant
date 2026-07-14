"""BGE Reranker provider (BAAI/bge-reranker-*) — a cross-encoder family trained
specifically for reranking, generally stronger than generic MS-MARCO cross-encoders
and multilingual (v2-m3), which matters for a bank operating across markets.

Prefers BAAI's own `FlagEmbedding` library (`FlagReranker.compute_score(...,
normalize=True)`, which applies the calibrated sigmoid the model was tuned with).
Falls back to loading the same checkpoint through `sentence_transformers.CrossEncoder`
if `FlagEmbedding` isn't installed — BGE reranker checkpoints are published in a
format that loads fine there too, just without FlagEmbedding's extras (e.g. its
optional layerwise/lightweight-head variants for the larger models).
"""

from __future__ import annotations

import asyncio
import threading

from backend.core.config import Settings
from backend.core.logging import get_logger
from backend.retrieval.reranking.types import RerankerProvider, sigmoid

logger = get_logger(__name__)


class BGERerankerProvider(RerankerProvider):
    name = "bge"

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.bge_reranker_model
        self._model = None
        self._backend: str | None = None  # "flag_embedding" | "cross_encoder"
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from FlagEmbedding import FlagReranker

                logger.info(
                    "loading_reranker_model",
                    extra={"provider": self.name, "backend": "flag_embedding"},
                )
                self._model = FlagReranker(self._model_name, use_fp16=True)
                self._backend = "flag_embedding"
            except ImportError:
                from sentence_transformers import CrossEncoder

                logger.warning(
                    "flagembedding_unavailable_falling_back_to_cross_encoder",
                    extra={"model": self._model_name},
                )
                self._model = CrossEncoder(self._model_name)
                self._backend = "cross_encoder"

    def _score_sync(self, query: str, documents: list[str]) -> list[float]:
        self._ensure_loaded()
        pairs = [[query, doc] for doc in documents]

        if self._backend == "flag_embedding":
            scores = self._model.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
            return [float(s) for s in scores]

        raw_scores = self._model.predict(pairs)
        return [sigmoid(float(s)) for s in raw_scores]

    async def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return await asyncio.to_thread(self._score_sync, query, documents)

    @property
    def is_available(self) -> bool:
        try:
            import FlagEmbedding  # noqa: F401
        except ImportError:
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                return False
        return True
