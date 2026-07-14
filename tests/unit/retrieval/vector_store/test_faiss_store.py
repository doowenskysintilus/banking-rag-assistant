import numpy as np
import pytest

from backend.core.config import Settings
from backend.retrieval.vector_store.faiss_store import FAISSVectorStore

DIM = 8


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        secret_key="x" * 16,
        faiss_index_dir=str(tmp_path / "faiss"),
        embedding_dim=DIM,
        **overrides,
    )


def _unit_vector(dim: int, hot_index: int) -> np.ndarray:
    vec = np.zeros(dim, dtype="float32")
    vec[hot_index] = 1.0
    return vec


def _vectors(*hot_indices: int) -> np.ndarray:
    return np.stack([_unit_vector(DIM, i) for i in hot_indices])


class TestIndexCreationAndAutoLoad:
    async def test_creates_a_new_empty_index_when_none_exists(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        results = await store.search(_unit_vector(DIM, 0), top_k=5)
        assert results == []

    async def test_automatically_loads_persisted_index_on_next_construction(
        self, tmp_path
    ):
        settings = _settings(tmp_path)
        store = FAISSVectorStore(settings)
        await store.add(
            [1, 2], _vectors(0, 1), metadatas=[{"owner_id": "u-1"}, {"owner_id": "u-2"}]
        )

        reloaded = FAISSVectorStore(settings)
        results = await reloaded.search(_unit_vector(DIM, 0), top_k=2)
        assert {r.vector_id for r in results} == {1, 2}

    async def test_supports_hnsw_index_factory(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path, faiss_index_factory="HNSW32"))
        await store.add([1, 2], _vectors(0, 1))
        results = await store.search(_unit_vector(DIM, 0), top_k=2)
        assert len(results) == 2


class TestAddAndSearch:
    async def test_search_ranks_the_closest_vector_first(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add([1, 2, 3], _vectors(0, 1, 2))

        results = await store.search(_unit_vector(DIM, 1), top_k=3)
        assert results[0].vector_id == 2

    async def test_search_returns_stored_metadata(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add(
            [1], _vectors(0), metadatas=[{"owner_id": "u-1", "document_id": "d-1"}]
        )

        results = await store.search(_unit_vector(DIM, 0), top_k=1)
        assert results[0].metadata == {"owner_id": "u-1", "document_id": "d-1"}

    async def test_search_on_empty_index_returns_empty_list(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        assert await store.search(_unit_vector(DIM, 0), top_k=5) == []

    async def test_top_k_limits_result_count(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add([1, 2, 3, 4], _vectors(0, 1, 2, 3))
        results = await store.search(_unit_vector(DIM, 0), top_k=2)
        assert len(results) == 2


class TestMetadataFiltering:
    async def test_filters_restrict_results_to_matching_metadata(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add(
            [1, 2, 3],
            _vectors(0, 0, 0),  # identical vectors — only metadata differs
            metadatas=[{"owner_id": "u-1"}, {"owner_id": "u-2"}, {"owner_id": "u-1"}],
        )

        results = await store.search(
            _unit_vector(DIM, 0), top_k=10, filters={"owner_id": "u-1"}
        )
        assert {r.vector_id for r in results} == {1, 3}

    async def test_selective_filter_still_returns_top_k_via_overfetch_retry(
        self, tmp_path
    ):
        # 20 vectors, only 2 match the filter, default top_k request of 2 — a naive
        # single-shot top_k*factor fetch (2*5=10) could miss both if they're not in
        # the nearest 10, so this exercises the retry-doubling behavior.
        store = FAISSVectorStore(
            _settings(
                tmp_path, faiss_filter_overfetch_factor=2, faiss_max_filter_retries=5
            )
        )
        vector_ids = list(range(20))
        vectors = np.stack([_unit_vector(DIM, i % DIM) for i in vector_ids])
        metadatas = [
            {"owner_id": "u-1" if i in (18, 19) else "u-2"} for i in vector_ids
        ]
        await store.add(vector_ids, vectors, metadatas=metadatas)

        results = await store.search(
            _unit_vector(DIM, 0), top_k=2, filters={"owner_id": "u-1"}
        )
        assert {r.vector_id for r in results} == {18, 19}

    async def test_filter_matching_nothing_returns_empty(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add([1], _vectors(0), metadatas=[{"owner_id": "u-1"}])
        results = await store.search(
            _unit_vector(DIM, 0), top_k=5, filters={"owner_id": "nobody"}
        )
        assert results == []


class TestDelete:
    async def test_deleted_vectors_no_longer_appear_in_search(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add([1, 2], _vectors(0, 1))
        await store.delete([1])

        results = await store.search(_unit_vector(DIM, 0), top_k=5)
        assert {r.vector_id for r in results} == {2}

    async def test_delete_removes_metadata_and_bm25_entries(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add(
            [1],
            _vectors(0),
            metadatas=[{"owner_id": "u-1"}],
            texts=["overdraft policy text"],
        )
        await store.delete([1])

        results = await store.hybrid_search(
            _unit_vector(DIM, 0), "overdraft policy", top_k=5
        )
        assert results == []


class TestUpdate:
    async def test_update_vector_changes_search_ranking(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add([1, 2], _vectors(0, 1))

        # Querying for dimension 1: vector 1 (at dim 0) doesn't match it yet.
        before = await store.search(_unit_vector(DIM, 1), top_k=2)
        score_before = next(r.score for r in before if r.vector_id == 1)
        assert score_before == pytest.approx(0.0, abs=1e-5)

        # Move vector 1 to point at dimension 1 instead of 0 — it should now match.
        await store.update([1], vectors=_vectors(1))

        after = await store.search(_unit_vector(DIM, 1), top_k=2)
        score_after = next(r.score for r in after if r.vector_id == 1)
        assert score_after == pytest.approx(1.0, abs=1e-5)

    async def test_update_metadata_only_leaves_vector_unchanged(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add([1], _vectors(0), metadatas=[{"owner_id": "u-1"}])

        await store.update([1], metadatas=[{"owner_id": "u-2"}])

        results = await store.search(_unit_vector(DIM, 0), top_k=1)
        assert results[0].metadata == {"owner_id": "u-2"}
        assert results[0].score == pytest.approx(
            1.0, abs=1e-5
        )  # vector itself untouched

    async def test_update_text_only_refreshes_the_lexical_index(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.add([1], _vectors(0), texts=["original overdraft text"])
        assert store._bm25.search(
            "overdraft", top_k=5
        )  # indexed under the original text

        await store.update([1], texts=["completely different vacation text"])

        assert store._bm25.search("overdraft", top_k=5) == []
        assert store._bm25.search("vacation", top_k=5)

    async def test_updating_unknown_id_is_skipped_without_raising(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        await store.update([999], metadatas=[{"owner_id": "u-1"}])  # must not raise
        assert await store.search(_unit_vector(DIM, 0), top_k=5) == []


class TestHybridSearch:
    async def test_lexical_match_can_outrank_dense_only_top_result(self, tmp_path):
        """A pure-dense top-1 search returns a decoy that's an exact vector match
        but shares no vocabulary with the query. Hybrid search's BM25 half should
        surface the "target" instead: it's dense-irrelevant (orthogonal vector) but
        an exact lexical match, and with 4 dense decoys tying for best dense rank
        (diluting each one's fusion contribution) the RRF math favors the target's
        unique top sparse rank. See the module docstring for the fusion formula.
        """
        store = FAISSVectorStore(_settings(tmp_path))

        decoy_ids = [1, 2, 3, 4]
        query_vector = _unit_vector(DIM, 0)
        await store.add(
            decoy_ids, np.stack([query_vector] * 4)
        )  # exact dense match, no text

        target_id = 5
        await store.add(
            [target_id],
            np.stack([_unit_vector(DIM, 7)]),  # orthogonal to the query vector
            texts=["overdraft covenant fees exact lexical match"],
        )

        dense_only = await store.search(query_vector, top_k=1)
        assert dense_only[0].vector_id in decoy_ids

        hybrid = await store.hybrid_search(
            query_vector, "overdraft covenant fees", top_k=1
        )
        assert hybrid[0].vector_id == target_id

    async def test_hybrid_search_respects_metadata_filters(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
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

    async def test_hybrid_search_on_empty_index_returns_empty(self, tmp_path):
        store = FAISSVectorStore(_settings(tmp_path))
        assert (
            await store.hybrid_search(_unit_vector(DIM, 0), "anything", top_k=5) == []
        )


class TestPersistence:
    async def test_save_persists_metadata_across_instances(self, tmp_path):
        settings = _settings(tmp_path)
        store = FAISSVectorStore(settings)
        await store.add(
            [1], _vectors(0), metadatas=[{"owner_id": "u-1"}], texts=["overdraft text"]
        )
        await store.save()

        reloaded = FAISSVectorStore(settings)
        results = await reloaded.hybrid_search(
            _unit_vector(DIM, 0), "overdraft", top_k=5
        )
        assert any(r.vector_id == 1 for r in results)

    async def test_no_temp_files_left_behind_after_persist(self, tmp_path):
        settings = _settings(tmp_path)
        store = FAISSVectorStore(settings)
        await store.add([1], _vectors(0))

        leftover = list((tmp_path / "faiss").glob("*.tmp"))
        assert leftover == []
