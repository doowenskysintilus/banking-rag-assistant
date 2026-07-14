import numpy as np
import pytest

from backend.core.config import Settings
from backend.core.exceptions import ExternalServiceError
from backend.retrieval.vector_store.pinecone_store import PineconeVectorStore
from tests.unit.retrieval.vector_store.pinecone_fakes import (
    FakePineconeProject,
    FakeServerlessSpec,
)

DIM = 8


def _settings(**overrides) -> Settings:
    return Settings(
        secret_key="x" * 16,
        pinecone_api_key="pk-fake",
        embedding_dim=DIM,
        **overrides,
    )


def _unit_vector(dim: int, hot_index: int) -> np.ndarray:
    vec = np.zeros(dim, dtype="float32")
    vec[hot_index] = 1.0
    return vec


def _vectors(*hot_indices: int) -> np.ndarray:
    return np.stack([_unit_vector(DIM, i) for i in hot_indices])


@pytest.fixture
def fake_project(monkeypatch) -> FakePineconeProject:
    import pinecone

    project = FakePineconeProject()
    monkeypatch.setattr(pinecone, "Pinecone", project.connect)
    monkeypatch.setattr(pinecone, "ServerlessSpec", FakeServerlessSpec)
    return project


def _build_store(fake_project, **settings_overrides) -> PineconeVectorStore:
    return PineconeVectorStore(_settings(**settings_overrides))


class TestConstruction:
    def test_raises_without_api_key(self):
        with pytest.raises(ExternalServiceError):
            PineconeVectorStore(Settings(secret_key="x" * 16, pinecone_api_key=""))

    def test_creates_index_when_missing(self, fake_project):
        store = _build_store(fake_project)
        assert store._index_name in fake_project.indexes
        assert fake_project.create_index_calls[0]["dimension"] == DIM

    def test_does_not_recreate_an_existing_index(self, fake_project):
        _build_store(fake_project, pinecone_index_name="banking-rag")
        # Second store instance — a fresh client "reconnecting" to the same
        # (fake) remote project — must not create the index again.
        PineconeVectorStore(_settings(pinecone_index_name="banking-rag"))
        assert len(fake_project.create_index_calls) == 1

    def test_skips_index_creation_when_disabled(self, fake_project):
        from tests.unit.retrieval.vector_store.pinecone_fakes import FakeIndex

        fake_project.indexes["banking-rag"] = FakeIndex(DIM)

        PineconeVectorStore(_settings(pinecone_create_index_if_missing=False))
        assert fake_project.create_index_calls == []


class TestAddAndSearch:
    async def test_search_ranks_closest_vector_first(self, fake_project):
        store = _build_store(fake_project)
        await store.add([1, 2, 3], _vectors(0, 1, 2))

        results = await store.search(_unit_vector(DIM, 1), top_k=3)
        assert results[0].vector_id == 2

    async def test_search_returns_stored_metadata(self, fake_project):
        store = _build_store(fake_project)
        await store.add([1], _vectors(0), metadatas=[{"owner_id": "u-1"}])

        results = await store.search(_unit_vector(DIM, 0), top_k=1)
        assert results[0].metadata == {"owner_id": "u-1"}

    async def test_internal_text_metadata_key_is_never_exposed(self, fake_project):
        store = _build_store(fake_project)
        await store.add(
            [1],
            _vectors(0),
            metadatas=[{"owner_id": "u-1"}],
            texts=["overdraft policy"],
        )

        results = await store.search(_unit_vector(DIM, 0), top_k=1)
        assert "_text" not in results[0].metadata

    async def test_search_on_empty_index_returns_empty(self, fake_project):
        store = _build_store(fake_project)
        assert await store.search(_unit_vector(DIM, 0), top_k=5) == []


class TestNativeMetadataFiltering:
    async def test_filter_is_passed_through_to_the_query_and_restricts_results(
        self, fake_project
    ):
        store = _build_store(fake_project)
        await store.add(
            [1, 2, 3],
            _vectors(0, 0, 0),
            metadatas=[{"owner_id": "u-1"}, {"owner_id": "u-2"}, {"owner_id": "u-1"}],
        )

        results = await store.search(
            _unit_vector(DIM, 0), top_k=10, filters={"owner_id": "u-1"}
        )
        assert {r.vector_id for r in results} == {1, 3}

    async def test_filter_matching_nothing_returns_empty(self, fake_project):
        store = _build_store(fake_project)
        await store.add([1], _vectors(0), metadatas=[{"owner_id": "u-1"}])
        results = await store.search(
            _unit_vector(DIM, 0), top_k=5, filters={"owner_id": "nobody"}
        )
        assert results == []


class TestDelete:
    async def test_deleted_vectors_no_longer_appear_in_search(self, fake_project):
        store = _build_store(fake_project)
        await store.add([1, 2], _vectors(0, 1))
        await store.delete([1])

        results = await store.search(_unit_vector(DIM, 0), top_k=5)
        assert {r.vector_id for r in results} == {2}

    async def test_delete_clears_local_bm25_entries(self, fake_project):
        store = _build_store(fake_project)
        await store.add([1], _vectors(0), texts=["overdraft policy text"])
        await store.delete([1])
        assert store._bm25.search("overdraft", top_k=5) == []


class TestUpdate:
    async def test_update_vector_changes_search_ranking(self, fake_project):
        store = _build_store(fake_project)
        await store.add([1, 2], _vectors(0, 1))

        before = await store.search(_unit_vector(DIM, 1), top_k=2)
        score_before = next(r.score for r in before if r.vector_id == 1)
        assert score_before == pytest.approx(0.0, abs=1e-5)

        await store.update([1], vectors=_vectors(1))

        after = await store.search(_unit_vector(DIM, 1), top_k=2)
        score_after = next(r.score for r in after if r.vector_id == 1)
        assert score_after == pytest.approx(1.0, abs=1e-5)

    async def test_update_metadata_only_leaves_vector_unchanged(self, fake_project):
        store = _build_store(fake_project)
        await store.add([1], _vectors(0), metadatas=[{"owner_id": "u-1"}])

        await store.update([1], metadatas=[{"tier": "premium"}])

        results = await store.search(_unit_vector(DIM, 0), top_k=1)
        # Pinecone's set_metadata merges rather than replaces.
        assert results[0].metadata == {"owner_id": "u-1", "tier": "premium"}
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_update_text_only_refreshes_the_lexical_index(self, fake_project):
        store = _build_store(fake_project)
        await store.add([1], _vectors(0), texts=["original overdraft text"])
        assert store._bm25.search("overdraft", top_k=5)

        await store.update([1], texts=["completely different vacation text"])

        assert store._bm25.search("overdraft", top_k=5) == []
        assert store._bm25.search("vacation", top_k=5)


class TestHybridSearch:
    async def test_lexical_match_can_outrank_dense_only_top_result(self, fake_project):
        store = _build_store(fake_project)

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

    async def test_hybrid_search_respects_metadata_filters(self, fake_project):
        store = _build_store(fake_project)
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

    async def test_hybrid_search_on_empty_index_returns_empty(self, fake_project):
        store = _build_store(fake_project)
        assert (
            await store.hybrid_search(_unit_vector(DIM, 0), "anything", top_k=5) == []
        )


class TestBM25Rebuild:
    async def test_new_instance_rebuilds_bm25_from_remote_metadata(self, fake_project):
        store = _build_store(fake_project, pinecone_index_name="rebuild-test")
        await store.add([1], _vectors(0), texts=["overdraft policy text"])

        # A second store instance (same fake remote index) should rebuild its
        # local BM25 index from the text stored in Pinecone's metadata.
        second_store = PineconeVectorStore(
            _settings(pinecone_index_name="rebuild-test")
        )
        assert second_store._bm25.search("overdraft", top_k=5)


class TestSave:
    async def test_save_is_a_no_op_and_does_not_raise(self, fake_project):
        store = _build_store(fake_project)
        await store.add([1], _vectors(0))
        await store.save()  # must not raise
