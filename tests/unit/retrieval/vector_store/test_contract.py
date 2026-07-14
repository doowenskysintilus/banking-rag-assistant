"""Strategy-pattern contract tests: the same behavior suite runs against both
`FAISSVectorStore` and `PineconeVectorStore` (mocked SDK) via one parametrized
`store` fixture. If a test in this file passes for one backend and fails for the
other, the two are no longer truly interchangeable behind `VectorStoreBase` — this
file exists to catch exactly that.

Backend-specific behavior (FAISS's `index_factory` choices, Pinecone's index
auto-creation, the BM25-rebuild-from-remote-metadata quirk) is covered separately
in `test_faiss_store.py` / `test_pinecone_store.py`, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.config import Settings
from backend.retrieval.vector_store.faiss_store import FAISSVectorStore
from backend.retrieval.vector_store.pinecone_store import PineconeVectorStore
from backend.retrieval.vector_store.types import VectorStoreBase
from tests.unit.retrieval.vector_store.pinecone_fakes import (
    FakePineconeProject,
    FakeServerlessSpec,
)

DIM = 8


def _unit_vector(dim: int, hot_index: int) -> np.ndarray:
    vec = np.zeros(dim, dtype="float32")
    vec[hot_index] = 1.0
    return vec


def _vectors(*hot_indices: int) -> np.ndarray:
    return np.stack([_unit_vector(DIM, i) for i in hot_indices])


def _make_faiss(tmp_path) -> VectorStoreBase:
    settings = Settings(
        secret_key="x" * 16, faiss_index_dir=str(tmp_path / "faiss"), embedding_dim=DIM
    )
    return FAISSVectorStore(settings)


def _make_pinecone(monkeypatch) -> VectorStoreBase:
    import pinecone

    project = FakePineconeProject()
    monkeypatch.setattr(pinecone, "Pinecone", project.connect)
    monkeypatch.setattr(pinecone, "ServerlessSpec", FakeServerlessSpec)
    settings = Settings(
        secret_key="x" * 16, pinecone_api_key="pk-fake", embedding_dim=DIM
    )
    return PineconeVectorStore(settings)


@pytest.fixture(params=["faiss", "pinecone"])
def store(request, tmp_path, monkeypatch) -> VectorStoreBase:
    if request.param == "faiss":
        return _make_faiss(tmp_path)
    return _make_pinecone(monkeypatch)


class TestSearchContract:
    async def test_empty_store_returns_no_results(self, store: VectorStoreBase):
        assert await store.search(_unit_vector(DIM, 0), top_k=5) == []

    async def test_search_finds_the_nearest_vector(self, store: VectorStoreBase):
        await store.add([1, 2, 3], _vectors(0, 1, 2))
        results = await store.search(_unit_vector(DIM, 1), top_k=1)
        assert results[0].vector_id == 2
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_metadata_round_trips_through_search(self, store: VectorStoreBase):
        await store.add(
            [1], _vectors(0), metadatas=[{"owner_id": "u-1", "document_id": "d-1"}]
        )
        results = await store.search(_unit_vector(DIM, 0), top_k=1)
        assert results[0].metadata == {"owner_id": "u-1", "document_id": "d-1"}

    async def test_filters_restrict_results_to_matching_metadata(
        self, store: VectorStoreBase
    ):
        await store.add(
            [1, 2, 3],
            _vectors(0, 0, 0),
            metadatas=[{"owner_id": "u-1"}, {"owner_id": "u-2"}, {"owner_id": "u-1"}],
        )
        results = await store.search(
            _unit_vector(DIM, 0), top_k=10, filters={"owner_id": "u-1"}
        )
        assert {r.vector_id for r in results} == {1, 3}

    async def test_filter_matching_nothing_returns_empty(self, store: VectorStoreBase):
        await store.add([1], _vectors(0), metadatas=[{"owner_id": "u-1"}])
        results = await store.search(
            _unit_vector(DIM, 0), top_k=5, filters={"owner_id": "nobody"}
        )
        assert results == []


class TestDeleteContract:
    async def test_deleted_vectors_no_longer_appear_in_search(
        self, store: VectorStoreBase
    ):
        await store.add([1, 2], _vectors(0, 1))
        await store.delete([1])
        results = await store.search(_unit_vector(DIM, 0), top_k=5)
        assert {r.vector_id for r in results} == {2}


class TestUpdateContract:
    async def test_update_vector_changes_search_ranking(self, store: VectorStoreBase):
        await store.add([1, 2], _vectors(0, 1))

        before = await store.search(_unit_vector(DIM, 1), top_k=2)
        assert next(r.score for r in before if r.vector_id == 1) == pytest.approx(
            0.0, abs=1e-5
        )

        await store.update([1], vectors=_vectors(1))

        after = await store.search(_unit_vector(DIM, 1), top_k=2)
        assert next(r.score for r in after if r.vector_id == 1) == pytest.approx(
            1.0, abs=1e-5
        )

    async def test_update_metadata_only_leaves_the_vector_unchanged(
        self, store: VectorStoreBase
    ):
        await store.add([1], _vectors(0), metadatas=[{"owner_id": "u-1"}])
        await store.update([1], metadatas=[{"owner_id": "u-1", "tier": "premium"}])

        results = await store.search(_unit_vector(DIM, 0), top_k=1)
        assert results[0].metadata["tier"] == "premium"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)


class TestHybridSearchContract:
    async def test_hybrid_search_on_empty_store_returns_empty(
        self, store: VectorStoreBase
    ):
        assert (
            await store.hybrid_search(_unit_vector(DIM, 0), "anything", top_k=5) == []
        )

    async def test_lexical_match_can_outrank_a_dense_only_top_result(
        self, store: VectorStoreBase
    ):
        decoy_ids = [1, 2, 3, 4]
        query_vector = _unit_vector(DIM, 0)
        await store.add(decoy_ids, np.stack([query_vector] * 4))

        target_id = 5
        await store.add(
            [target_id],
            np.stack([_unit_vector(DIM, 7)]),
            texts=["overdraft covenant fees exact lexical match"],
        )

        dense_only = await store.search(query_vector, top_k=1)
        assert dense_only[0].vector_id in decoy_ids

        hybrid = await store.hybrid_search(
            query_vector, "overdraft covenant fees", top_k=1
        )
        assert hybrid[0].vector_id == target_id

    async def test_hybrid_search_respects_metadata_filters(
        self, store: VectorStoreBase
    ):
        await store.add(
            [1, 2],
            _vectors(0, 0),
            metadatas=[{"owner_id": "u-1"}, {"owner_id": "u-2"}],
            texts=["overdraft policy", "overdraft policy"],
        )
        results = await store.hybrid_search(
            _unit_vector(DIM, 0),
            "overdraft policy",
            top_k=5,
            filters={"owner_id": "u-1"},
        )
        assert {r.vector_id for r in results} == {1}


class TestSaveContract:
    async def test_save_does_not_raise(self, store: VectorStoreBase):
        await store.add([1], _vectors(0))
        await store.save()
