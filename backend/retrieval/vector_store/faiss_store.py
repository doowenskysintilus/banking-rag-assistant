"""FAISS-backed vector store: index lifecycle (create/save/load), CRUD, metadata
filtering, and dense+lexical hybrid search.

Suitable for single-instance deployments; for multi-instance production use, front
this with a shared volume or switch to the Pinecone implementation
(`backend.retrieval.vector_store.pinecone_store`).
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError
from backend.core.logging import get_logger
from backend.retrieval.vector_store.bm25 import BM25Index
from backend.retrieval.vector_store.filters import matches
from backend.retrieval.vector_store.hybrid import reciprocal_rank_fusion
from backend.retrieval.vector_store.metadata_store import MetadataStore
from backend.retrieval.vector_store.types import Filter, SearchResult, VectorStoreBase

logger = get_logger(__name__)


class FAISSVectorStore(VectorStoreBase):
    def __init__(self, settings: Settings) -> None:
        import faiss  # imported lazily so the dependency is optional at import time

        self._faiss = faiss
        self._dim = settings.embedding_dim
        self._index_factory_spec = settings.faiss_index_factory
        self._overfetch_factor = max(1, settings.faiss_filter_overfetch_factor)
        self._max_filter_retries = max(1, settings.faiss_max_filter_retries)
        self._rrf_k = settings.faiss_hybrid_rrf_k

        self._index_dir = Path(settings.faiss_index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_dir / "index.faiss"
        self._metadata_path = self._index_dir / "metadata.json"

        self._lock = threading.Lock()
        self._index = self._load_or_create_index()
        self._metadata = MetadataStore(self._metadata_path)
        self._bm25 = self._rebuild_bm25()

    # --- index lifecycle ---------------------------------------------------

    def _create_index(self):
        base_index = self._faiss.index_factory(
            self._dim, self._index_factory_spec, self._faiss.METRIC_INNER_PRODUCT
        )
        return self._faiss.IndexIDMap2(base_index)

    def _load_or_create_index(self):
        if self._index_path.exists():
            logger.info("faiss_index_loaded", extra={"path": str(self._index_path)})
            return self._faiss.read_index(str(self._index_path))
        logger.info(
            "faiss_index_created",
            extra={"factory": self._index_factory_spec, "dim": self._dim},
        )
        return self._create_index()

    def _rebuild_bm25(self) -> BM25Index:
        """BM25 state isn't persisted separately — it's cheap to rebuild from the
        texts already stored in the metadata sidecar on every load.
        """
        bm25 = BM25Index()
        for vector_id in self._metadata.known_ids():
            text = self._metadata.text_of(vector_id)
            if text:
                bm25.add(vector_id, text)
        return bm25

    def _persist(self) -> None:
        tmp_path = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        self._faiss.write_index(self._index, str(tmp_path))
        tmp_path.replace(self._index_path)
        self._metadata.save()

    def _ensure_trained(self, vectors: np.ndarray) -> None:
        if not self._index.is_trained:
            if len(vectors) == 0:
                return
            self._index.train(vectors)

    # --- mutation (sync, run under lock via to_thread) ----------------------

    def _add_sync(
        self,
        vector_ids: list[int],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]] | None,
        texts: list[str] | None,
    ) -> None:
        with self._lock:
            self._ensure_trained(vectors)
            ids = np.asarray(vector_ids, dtype="int64")
            self._index.add_with_ids(vectors, ids)

            for i, vector_id in enumerate(vector_ids):
                metadata = metadatas[i] if metadatas else {}
                text = texts[i] if texts else None
                self._metadata.set(vector_id, metadata, text)
                if text:
                    self._bm25.add(vector_id, text)

            self._persist()

    def _update_sync(
        self,
        vector_ids: list[int],
        vectors: np.ndarray | None,
        metadatas: list[dict[str, Any]] | None,
        texts: list[str] | None,
    ) -> None:
        with self._lock:
            for i, vector_id in enumerate(vector_ids):
                existing = self._metadata.get(vector_id)
                if existing is None:
                    logger.warning(
                        "vector_update_skipped_unknown_id",
                        extra={"vector_id": vector_id},
                    )
                    continue

                if vectors is not None:
                    id_arr = np.asarray([vector_id], dtype="int64")
                    self._index.remove_ids(id_arr)
                    self._ensure_trained(vectors[i : i + 1])
                    self._index.add_with_ids(vectors[i : i + 1], id_arr)

                new_metadata = (
                    metadatas[i] if metadatas is not None else existing["metadata"]
                )
                new_text = texts[i] if texts is not None else existing["text"]

                if existing["text"]:
                    self._bm25.remove(vector_id)
                self._metadata.set(vector_id, new_metadata, new_text)
                if new_text:
                    self._bm25.add(vector_id, new_text)

            self._persist()

    def _delete_sync(self, vector_ids: list[int]) -> None:
        with self._lock:
            ids = np.asarray(vector_ids, dtype="int64")
            self._index.remove_ids(ids)
            for vector_id in vector_ids:
                self._metadata.delete(vector_id)
                self._bm25.remove(vector_id)
            self._persist()

    # --- search (sync, run under lock via to_thread) ------------------------

    def _dense_candidates(
        self, query_vector: np.ndarray, fetch_k: int
    ) -> list[tuple[int, float]]:
        scores, ids = self._index.search(query_vector.reshape(1, -1), fetch_k)
        return [
            (int(vid), float(score))
            for vid, score in zip(ids[0], scores[0], strict=True)
            if vid != -1
        ]

    def _hybrid_candidates(
        self, query_vector: np.ndarray, query_text: str, fetch_k: int
    ) -> list[tuple[int, float]]:
        dense_ranking = [
            vid for vid, _ in self._dense_candidates(query_vector, fetch_k)
        ]
        sparse_ranking = [vid for vid, _ in self._bm25.search(query_text, fetch_k)]
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=self._rrf_k)
        return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)

    def _fetch_filtered(
        self,
        fetch_fn: Callable[[int], list[tuple[int, float]]],
        top_k: int,
        filters: Filter | None,
        *,
        min_fetch_k: int = 0,
    ) -> list[tuple[int, float]]:
        """Runs `fetch_fn(fetch_k)` and applies `filters`, growing `fetch_k`
        (doubling, capped at index size) when the filter excludes too many
        candidates to reach `top_k` — bounded by `_max_filter_retries` attempts.

        `min_fetch_k` lets a caller force a wider candidate pool even without a
        filter — hybrid search needs this: fusing two rankings that were each
        truncated to the bare `top_k` starves Reciprocal Rank Fusion of the
        candidates it needs to work with (a lexical-only match can't out-rank a
        dense-only match if it was never fetched in the first place).
        """
        total = self._index.ntotal
        if total == 0:
            return []

        base_fetch_k = top_k if not filters else top_k * self._overfetch_factor
        fetch_k = min(total, max(base_fetch_k, min_fetch_k))
        candidates: list[tuple[int, float]] = []

        for attempt in range(self._max_filter_retries):
            raw_candidates = fetch_fn(fetch_k)
            if filters:
                candidates = [
                    (vector_id, score)
                    for vector_id, score in raw_candidates
                    if matches(self._metadata.metadata_of(vector_id), filters)
                ]
            else:
                candidates = raw_candidates

            if len(candidates) >= top_k or fetch_k >= total:
                break
            fetch_k = min(total, fetch_k * 2)

        return candidates[:top_k]

    def _to_results(self, candidates: list[tuple[int, float]]) -> list[SearchResult]:
        return [
            SearchResult(
                vector_id=vector_id,
                score=score,
                metadata=self._metadata.metadata_of(vector_id),
            )
            for vector_id, score in candidates
        ]

    def _search_sync(
        self, query_vector: np.ndarray, top_k: int, filters: Filter | None
    ) -> list[SearchResult]:
        with self._lock:
            candidates = self._fetch_filtered(
                lambda k: self._dense_candidates(query_vector, k), top_k, filters
            )
            return self._to_results(candidates)

    def _hybrid_search_sync(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int,
        filters: Filter | None,
    ) -> list[SearchResult]:
        with self._lock:
            candidates = self._fetch_filtered(
                lambda k: self._hybrid_candidates(query_vector, query_text, k),
                top_k,
                filters,
                min_fetch_k=top_k * self._overfetch_factor,
            )
            return self._to_results(candidates)

    # --- public async API ----------------------------------------------------

    async def add(
        self,
        vector_ids: list[int],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
        texts: list[str] | None = None,
    ) -> None:
        try:
            await asyncio.to_thread(
                self._add_sync, vector_ids, vectors, metadatas, texts
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("faiss_add_failed")
            raise ExternalServiceError("Failed to index document embeddings.") from exc

    async def update(
        self,
        vector_ids: list[int],
        vectors: np.ndarray | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        texts: list[str] | None = None,
    ) -> None:
        try:
            await asyncio.to_thread(
                self._update_sync, vector_ids, vectors, metadatas, texts
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("faiss_update_failed")
            raise ExternalServiceError("Failed to update document embeddings.") from exc

    async def delete(self, vector_ids: list[int]) -> None:
        try:
            await asyncio.to_thread(self._delete_sync, vector_ids)
        except Exception as exc:  # noqa: BLE001
            logger.exception("faiss_delete_failed")
            raise ExternalServiceError("Failed to remove document embeddings.") from exc

    async def search(
        self, query_vector: np.ndarray, top_k: int, filters: Filter | None = None
    ) -> list[SearchResult]:
        try:
            return await asyncio.to_thread(
                self._search_sync, query_vector, top_k, filters
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("faiss_search_failed")
            raise ExternalServiceError("Failed to search the knowledge base.") from exc

    async def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchResult]:
        try:
            return await asyncio.to_thread(
                self._hybrid_search_sync, query_vector, query_text, top_k, filters
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("faiss_hybrid_search_failed")
            raise ExternalServiceError("Failed to search the knowledge base.") from exc

    async def save(self) -> None:
        try:
            await asyncio.to_thread(self._persist)
        except Exception as exc:  # noqa: BLE001
            logger.exception("faiss_save_failed")
            raise ExternalServiceError(
                "Failed to persist the knowledge base index."
            ) from exc
