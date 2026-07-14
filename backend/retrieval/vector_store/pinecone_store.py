"""Pinecone-backed implementation of `VectorStoreBase` — the managed-cloud
counterpart to `FAISSVectorStore`. Same Strategy interface, same behavioral
contract (see `tests/unit/retrieval/vector_store/test_contract.py`), different
tradeoffs:

  * Vectors and metadata live in Pinecone, not on local disk — `save()` is a
    no-op (every `add`/`update`/`delete` is already durable server-side).
  * Metadata filtering is native (Pinecone's query `filter=` parameter uses the
    same Mongo-style operator syntax as `backend.retrieval.vector_store.filters`
    — no translation layer needed), so `search()` doesn't need FAISS's
    over-fetch-and-post-filter dance.
  * Pinecone has no built-in lexical/BM25 search reachable from a plain dense
    index, so `hybrid_search()` fuses Pinecone's dense query with the *same*
    local `BM25Index` FAISS uses. That local index is rebuilt at startup by
    paging through the remote index, bounded by
    `settings.pinecone_bm25_rebuild_limit` — for very large corpora this is
    necessarily a partial rebuild; prefer FAISS (whose BM25 rebuild is exact,
    from a local metadata file) if exhaustive lexical recall matters at scale.

Pinecone vector IDs are strings; this store's public interface uses ints (to
match FAISS and the rest of the app), so ids are converted at the boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError
from backend.core.logging import get_logger
from backend.retrieval.vector_store.bm25 import BM25Index
from backend.retrieval.vector_store.filters import matches
from backend.retrieval.vector_store.hybrid import reciprocal_rank_fusion
from backend.retrieval.vector_store.types import Filter, SearchResult, VectorStoreBase

logger = get_logger(__name__)

# Reserved metadata key used to smuggle the chunk's original text through Pinecone
# (needed to rebuild the local BM25 index) without it leaking into the metadata
# dict callers see back in `SearchResult.metadata`.
_TEXT_METADATA_KEY = "_text"


class PineconeVectorStore(VectorStoreBase):
    def __init__(self, settings: Settings) -> None:
        if not settings.pinecone_api_key:
            raise ExternalServiceError(
                "PINECONE_API_KEY is required when VECTOR_STORE_PROVIDER=pinecone."
            )

        from pinecone import Pinecone, ServerlessSpec

        self._serverless_spec_cls = ServerlessSpec
        self._client = Pinecone(api_key=settings.pinecone_api_key)

        self._index_name = settings.pinecone_index_name
        self._namespace = settings.pinecone_namespace or None
        self._dim = settings.embedding_dim
        self._metric = settings.pinecone_metric
        self._cloud = settings.pinecone_cloud
        self._region = settings.pinecone_region
        self._overfetch_factor = max(1, settings.faiss_filter_overfetch_factor)
        self._rrf_k = settings.faiss_hybrid_rrf_k
        self._bm25_rebuild_limit = settings.pinecone_bm25_rebuild_limit

        if settings.pinecone_create_index_if_missing:
            self._ensure_index_exists()

        self._index = self._client.Index(self._index_name)
        self._bm25 = BM25Index()
        self._rebuild_bm25_best_effort()

    # --- index lifecycle -----------------------------------------------------

    def _existing_index_names(self) -> set[str]:
        indexes = self._client.list_indexes()
        if hasattr(indexes, "names"):
            return set(indexes.names())
        return {
            item["name"] if isinstance(item, dict) else item.name for item in indexes
        }

    def _ensure_index_exists(self) -> None:
        if self._index_name in self._existing_index_names():
            return
        logger.info(
            "pinecone_index_created",
            extra={"index": self._index_name, "dim": self._dim},
        )
        self._client.create_index(
            name=self._index_name,
            dimension=self._dim,
            metric=self._metric,
            spec=self._serverless_spec_cls(cloud=self._cloud, region=self._region),
        )

    def _rebuild_bm25_best_effort(self) -> None:
        """Best-effort: page through the remote index and re-populate the local
        BM25 index from each vector's stored text. Never raises — a failure here
        just means hybrid_search falls back to dense-only quality until the
        process restarts with better luck (or vectors get re-added/updated).
        """
        try:
            count = 0
            for vector_id, text in self._iter_remote_texts():
                if count >= self._bm25_rebuild_limit:
                    logger.warning(
                        "pinecone_bm25_rebuild_truncated",
                        extra={"limit": self._bm25_rebuild_limit},
                    )
                    break
                self._bm25.add(vector_id, text)
                count += 1
        except Exception:  # noqa: BLE001 - degrade to dense-only hybrid search, never fail startup
            logger.warning("pinecone_bm25_rebuild_failed")

    def _iter_remote_texts(self):
        for id_batch in self._index.list(namespace=self._namespace):
            fetch_response = self._index.fetch(
                ids=list(id_batch), namespace=self._namespace
            )
            for vector_id_str, record in fetch_response.vectors.items():
                text = (record.metadata or {}).get(_TEXT_METADATA_KEY)
                if text:
                    yield int(vector_id_str), text

    # --- helpers ---------------------------------------------------------------

    @staticmethod
    def _strip_internal_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        clean = dict(metadata or {})
        clean.pop(_TEXT_METADATA_KEY, None)
        return clean

    def _fetch_metadata_map(self, vector_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not vector_ids:
            return {}
        response = self._index.fetch(
            ids=[str(v) for v in vector_ids], namespace=self._namespace
        )
        return {
            int(vid_str): dict(record.metadata or {})
            for vid_str, record in response.vectors.items()
        }

    # --- mutation --------------------------------------------------------------

    def _add_sync(
        self,
        vector_ids: list[int],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]] | None,
        texts: list[str] | None,
    ) -> None:
        items = []
        for i, vector_id in enumerate(vector_ids):
            metadata = dict(metadatas[i]) if metadatas else {}
            text = texts[i] if texts else None
            if text:
                metadata[_TEXT_METADATA_KEY] = text
            items.append(
                {
                    "id": str(vector_id),
                    "values": vectors[i].tolist(),
                    "metadata": metadata,
                }
            )

        self._index.upsert(vectors=items, namespace=self._namespace)

        for i, vector_id in enumerate(vector_ids):
            text = texts[i] if texts else None
            if text:
                self._bm25.add(vector_id, text)

    def _update_sync(
        self,
        vector_ids: list[int],
        vectors: np.ndarray | None,
        metadatas: list[dict[str, Any]] | None,
        texts: list[str] | None,
    ) -> None:
        for i, vector_id in enumerate(vector_ids):
            vector_id_str = str(vector_id)
            text = texts[i] if texts is not None else None

            if vectors is not None:
                metadata = dict(metadatas[i]) if metadatas is not None else {}
                if text:
                    metadata[_TEXT_METADATA_KEY] = text
                self._index.upsert(
                    vectors=[
                        {
                            "id": vector_id_str,
                            "values": vectors[i].tolist(),
                            "metadata": metadata,
                        }
                    ],
                    namespace=self._namespace,
                )
            elif metadatas is not None or texts is not None:
                # Pinecone's `set_metadata` merges into existing metadata server-side
                # rather than replacing it wholesale, so we don't need to fetch the
                # current metadata first just to preserve untouched fields.
                set_metadata = dict(metadatas[i]) if metadatas is not None else {}
                if texts is not None:
                    set_metadata[_TEXT_METADATA_KEY] = text or ""
                self._index.update(
                    id=vector_id_str,
                    set_metadata=set_metadata,
                    namespace=self._namespace,
                )

            if texts is not None:
                if text:
                    self._bm25.add(vector_id, text)
                else:
                    self._bm25.remove(vector_id)

    def _delete_sync(self, vector_ids: list[int]) -> None:
        self._index.delete(ids=[str(v) for v in vector_ids], namespace=self._namespace)
        for vector_id in vector_ids:
            self._bm25.remove(vector_id)

    # --- search ------------------------------------------------------------------

    def _search_sync(
        self, query_vector: np.ndarray, top_k: int, filters: Filter | None
    ) -> list[SearchResult]:
        response = self._index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            filter=dict(filters) if filters else None,
            include_metadata=True,
            namespace=self._namespace,
        )
        return [
            SearchResult(
                vector_id=int(match.id),
                score=float(match.score),
                metadata=self._strip_internal_metadata(match.metadata),
            )
            for match in response.matches
        ]

    def _hybrid_search_sync(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int,
        filters: Filter | None,
    ) -> list[SearchResult]:
        fetch_k = max(top_k * self._overfetch_factor, top_k)

        dense_response = self._index.query(
            vector=query_vector.tolist(),
            top_k=fetch_k,
            filter=dict(filters) if filters else None,
            include_metadata=True,
            namespace=self._namespace,
        )
        # Pinecone already applied `filters` server-side for these.
        dense_metadata = {
            int(m.id): dict(m.metadata or {}) for m in dense_response.matches
        }
        dense_ranking = list(dense_metadata.keys())

        sparse_candidates = self._bm25.search(query_text, fetch_k)
        sparse_ids = [vector_id for vector_id, _ in sparse_candidates]

        # BM25 knows nothing about Pinecone metadata, so any sparse candidate not
        # already confirmed by the (filtered) dense query needs an explicit check.
        ids_needing_check = [vid for vid in sparse_ids if vid not in dense_metadata]
        extra_metadata: dict[int, dict[str, Any]] = {}
        if filters and ids_needing_check:
            fetched = self._fetch_metadata_map(ids_needing_check)
            extra_metadata = {
                vid: meta for vid, meta in fetched.items() if matches(meta, filters)
            }
        elif ids_needing_check:
            extra_metadata = self._fetch_metadata_map(ids_needing_check)

        if filters:
            sparse_ranking = [
                vid
                for vid in sparse_ids
                if vid in dense_metadata or vid in extra_metadata
            ]
        else:
            sparse_ranking = sparse_ids

        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=self._rrf_k)
        ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:top_k]

        results = []
        for vector_id, score in ranked:
            metadata = (
                dense_metadata.get(vector_id) or extra_metadata.get(vector_id) or {}
            )
            results.append(
                SearchResult(
                    vector_id=vector_id,
                    score=score,
                    metadata=self._strip_internal_metadata(metadata),
                )
            )
        return results

    # --- public async API (Pinecone's client is sync, so no thread offload is
    # strictly required for CPU reasons — kept `async def` purely to satisfy the
    # shared `VectorStoreBase` interface FAISS also implements) -------------------

    async def add(
        self,
        vector_ids: list[int],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
        texts: list[str] | None = None,
    ) -> None:
        try:
            self._add_sync(vector_ids, vectors, metadatas, texts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("pinecone_add_failed")
            raise ExternalServiceError("Failed to index document embeddings.") from exc

    async def update(
        self,
        vector_ids: list[int],
        vectors: np.ndarray | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        texts: list[str] | None = None,
    ) -> None:
        try:
            self._update_sync(vector_ids, vectors, metadatas, texts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("pinecone_update_failed")
            raise ExternalServiceError("Failed to update document embeddings.") from exc

    async def delete(self, vector_ids: list[int]) -> None:
        try:
            self._delete_sync(vector_ids)
        except Exception as exc:  # noqa: BLE001
            logger.exception("pinecone_delete_failed")
            raise ExternalServiceError("Failed to remove document embeddings.") from exc

    async def search(
        self, query_vector: np.ndarray, top_k: int, filters: Filter | None = None
    ) -> list[SearchResult]:
        try:
            return self._search_sync(query_vector, top_k, filters)
        except Exception as exc:  # noqa: BLE001
            logger.exception("pinecone_search_failed")
            raise ExternalServiceError("Failed to search the knowledge base.") from exc

    async def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchResult]:
        try:
            return self._hybrid_search_sync(query_vector, query_text, top_k, filters)
        except Exception as exc:  # noqa: BLE001
            logger.exception("pinecone_hybrid_search_failed")
            raise ExternalServiceError("Failed to search the knowledge base.") from exc

    async def save(self) -> None:
        # Every mutation above is already durably written to Pinecone — nothing
        # local to flush. Kept as a real (if trivial) method so callers can treat
        # every VectorStoreBase implementation identically.
        return None
