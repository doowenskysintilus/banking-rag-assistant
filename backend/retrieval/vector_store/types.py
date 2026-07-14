"""Shared types for the vector store module."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# A metadata filter is a small Mongo-style operator dict, e.g.:
#   {"owner_id": "u-123"}
#   {"language": {"$in": ["en", "fr"]}}
#   {"$and": [{"owner_id": "u-123"}, {"document_id": {"$ne": "d-1"}}]}
# See `backend.retrieval.vector_store.filters.matches` for the full semantics.
Filter = dict[str, Any]


@dataclass
class SearchResult:
    vector_id: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStoreBase(abc.ABC):
    @abc.abstractmethod
    async def add(
        self,
        vector_ids: list[int],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
        texts: list[str] | None = None,
    ) -> None:
        """Add new vectors. `metadatas`/`texts`, if given, must be the same length
        as `vector_ids`. `texts` (the original chunk text) powers lexical/hybrid
        search — omit it for vectors that should only ever be found by dense search.
        """

    @abc.abstractmethod
    async def update(
        self,
        vector_ids: list[int],
        vectors: np.ndarray | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        texts: list[str] | None = None,
    ) -> None:
        """Update existing vectors/metadata/text in place. Any of `vectors`,
        `metadatas`, `texts` may be omitted (None) to leave that aspect unchanged.
        Unknown vector_ids are skipped (logged), not raised.
        """

    @abc.abstractmethod
    async def delete(self, vector_ids: list[int]) -> None: ...

    @abc.abstractmethod
    async def search(
        self, query_vector: np.ndarray, top_k: int, filters: Filter | None = None
    ) -> list[SearchResult]:
        """Dense (cosine/inner-product) search, optionally restricted to vectors
        whose metadata matches `filters`.
        """

    @abc.abstractmethod
    async def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchResult]:
        """Dense + lexical (BM25) search fused by Reciprocal Rank Fusion. `.score`
        on the results is an RRF score, not a cosine similarity — useful for
        ranking, not for absolute-similarity thresholds.
        """

    @abc.abstractmethod
    async def save(self) -> None:
        """Explicitly flush to disk. Mutating methods already persist after every
        call — this is for callers that want to force a sync point (e.g. before
        a graceful shutdown).
        """
