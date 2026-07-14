import sys
import types

import numpy as np

from backend.core.config import Settings
from backend.retrieval.enterprise.service import EnterpriseRetrievalService
from tests.unit.retrieval.enterprise.conftest import (
    FakeEmbeddingService,
    FakeVectorStore,
    iso_days_ago,
    make_chunk,
    make_document,
    make_user,
)

DIM = 4


def _settings(**overrides) -> Settings:
    defaults: dict = {
        "secret_key": "x" * 16,
        "enterprise_dense_top_k": 10,
        "enterprise_bm25_top_k": 10,
        "enterprise_ensemble_weights": (0.5, 0.5),
        "enterprise_multi_query_enabled": False,
        "enterprise_reranking_enabled": False,
        "enterprise_intelligent_ranking_enabled": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class ExactMapEmbeddingService:
    """Test double: returns a hand-picked vector for known texts (queries), and
    the zero vector for anything else — lets a test control dense similarity
    precisely, the way `FakeEmbeddingService`'s hash-based encoding can't.
    """

    def __init__(self, mapping: dict[str, np.ndarray], dim: int = DIM) -> None:
        self._mapping = mapping
        self.dim = dim

    async def embed_query(self, text: str) -> np.ndarray:
        return self._mapping.get(text, np.zeros(self.dim, dtype="float32"))

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([await self.embed_query(t) for t in texts])


def _unit_vector(hot_index: int) -> np.ndarray:
    vec = np.zeros(DIM, dtype="float32")
    vec[hot_index] = 1.0
    return vec


def _install_fake_cross_encoder(monkeypatch, predict_fn) -> None:
    fake_module = types.ModuleType("sentence_transformers")

    class FakeCrossEncoder:
        def __init__(self, model_name):
            pass

        def predict(self, pairs):
            return predict_fn(pairs)

    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


class TestBasicRetrieval:
    async def test_finds_the_relevant_chunk(self, db_session):
        owner = await make_user(db_session)
        doc = await make_document(db_session, owner_id=owner.id)
        chunk = await make_chunk(
            db_session, document=doc, content="overdraft policy details", vector_id=1
        )
        await db_session.commit()

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        vector = await embeddings.embed_query("overdraft policy details")
        await vector_store.add(
            [1], np.stack([vector]), metadatas=[{"owner_id": owner.id}]
        )

        service = EnterpriseRetrievalService(embeddings, vector_store, _settings())
        results = await service.retrieve(
            db_session, "overdraft policy details", owner.id
        )

        assert len(results) == 1
        assert results[0].chunk_id == chunk.id
        assert results[0].content == "overdraft policy details"

    async def test_top_k_truncates_results(self, db_session):
        owner = await make_user(db_session)
        doc = await make_document(db_session, owner_id=owner.id)
        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)

        vectors = []
        for i in range(5):
            content = f"policy clause number {i}"
            await make_chunk(db_session, document=doc, content=content, vector_id=i)
            vectors.append(await embeddings.embed_query(content))
        await db_session.commit()
        await vector_store.add(
            list(range(5)), np.stack(vectors), metadatas=[{"owner_id": owner.id}] * 5
        )

        service = EnterpriseRetrievalService(embeddings, vector_store, _settings())
        results = await service.retrieve(db_session, "policy clause", owner.id, top_k=2)

        assert len(results) == 2


class TestOwnerIsolation:
    async def test_never_returns_another_owners_chunks(self, db_session):
        owner = await make_user(db_session)
        attacker = await make_user(db_session)

        victim_doc = await make_document(db_session, owner_id=owner.id)
        await make_chunk(
            db_session,
            document=victim_doc,
            content="victim confidential balance",
            vector_id=1,
        )
        await db_session.commit()

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        vector = await embeddings.embed_query("victim confidential balance")
        await vector_store.add(
            [1], np.stack([vector]), metadatas=[{"owner_id": owner.id}]
        )

        service = EnterpriseRetrievalService(embeddings, vector_store, _settings())
        results = await service.retrieve(
            db_session, "victim confidential balance", attacker.id
        )

        assert results == []


class TestMetadataFiltering:
    async def test_extra_filters_are_applied_on_top_of_owner_scoping(self, db_session):
        owner = await make_user(db_session)
        doc = await make_document(db_session, owner_id=owner.id)
        await make_chunk(
            db_session, document=doc, content="english overdraft policy", vector_id=1
        )
        await make_chunk(
            db_session, document=doc, content="french overdraft policy", vector_id=2
        )
        await db_session.commit()

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        vec1 = await embeddings.embed_query("english overdraft policy")
        vec2 = await embeddings.embed_query("french overdraft policy")
        await vector_store.add(
            [1, 2],
            np.stack([vec1, vec2]),
            metadatas=[
                {"owner_id": owner.id, "language": "en"},
                {"owner_id": owner.id, "language": "fr"},
            ],
        )

        service = EnterpriseRetrievalService(embeddings, vector_store, _settings())
        results = await service.retrieve(
            db_session, "overdraft policy", owner.id, filters={"language": "en"}
        )

        assert {r.chunk_id for r in results} != set()
        for result in results:
            assert result.content == "english overdraft policy"


class TestHybridFusion:
    async def test_lexical_match_can_outrank_a_dense_only_top_result(self, db_session):
        owner = await make_user(db_session)
        doc = await make_document(db_session, owner_id=owner.id)

        query_text = "overdraft covenant fees"
        query_vector = _unit_vector(0)

        decoy_texts = [f"unrelated filler content number {i}" for i in range(4)]
        for i, text in enumerate(decoy_texts):
            await make_chunk(db_session, document=doc, content=text, vector_id=i)

        target_text = "overdraft covenant fees exact lexical match"
        await make_chunk(db_session, document=doc, content=target_text, vector_id=4)
        await db_session.commit()

        vector_store = FakeVectorStore()
        decoy_vectors = np.stack(
            [query_vector] * 4
        )  # exact dense match, no lexical overlap
        target_vector = _unit_vector(3)  # orthogonal — dense-irrelevant
        await vector_store.add(
            list(range(4)), decoy_vectors, metadatas=[{"owner_id": owner.id}] * 4
        )
        await vector_store.add(
            [4], np.stack([target_vector]), metadatas=[{"owner_id": owner.id}]
        )

        embeddings = ExactMapEmbeddingService({query_text: query_vector})

        # Sanity check: pure dense search alone would surface a decoy (all four
        # tie for an exact vector match; the lexical-only target scores 0 there).
        dense_only_hits = await vector_store.search(query_vector, top_k=1)
        assert dense_only_hits[0].vector_id in range(4)

        # With equal 0.5/0.5 weights this scenario is a perfect structural RRF
        # tie: the decoy that wins dense rank 1 also loses sparse rank worst (by
        # the same index-order tie-break BM25 and the fake vector store both use
        # for identically-scored candidates), landing it at the exact same fused
        # score as the target. Weighting BM25 slightly higher — a legitimate,
        # realistic choice for exact-term-heavy domains like banking/legal text
        # (see backend.ingestion.chunking's rationale for the same trade-off) —
        # breaks the tie deterministically instead of leaving it to sort stability.
        settings = _settings(enterprise_ensemble_weights=(0.3, 0.7))
        service = EnterpriseRetrievalService(embeddings, vector_store, settings)
        results = await service.retrieve(db_session, query_text, owner.id, top_k=1)
        assert results[0].content == target_text


class TestReranking:
    async def test_reranker_reorders_the_fused_candidates(
        self, db_session, monkeypatch
    ):
        owner = await make_user(db_session)
        doc = await make_document(db_session, owner_id=owner.id)
        await make_chunk(
            db_session, document=doc, content="dense favorite", vector_id=1
        )
        await make_chunk(
            db_session, document=doc, content="reranker favorite", vector_id=2
        )
        await db_session.commit()

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        query_vector = await embeddings.embed_query("query")
        # vector_id 1 is the closer dense match; the fake cross-encoder below will
        # prefer vector_id 2 instead.
        await vector_store.add(
            [1, 2],
            np.stack([query_vector, np.zeros(DIM, dtype="float32")]),
            metadatas=[{"owner_id": owner.id}] * 2,
        )

        def predict_fn(pairs):
            return [1.0 if "reranker favorite" in text else 0.0 for _, text in pairs]

        _install_fake_cross_encoder(monkeypatch, predict_fn)

        settings = _settings(
            enterprise_reranking_enabled=True,
            enterprise_reranker_provider="cross_encoder",
            enterprise_rerank_top_n=2,
        )
        service = EnterpriseRetrievalService(embeddings, vector_store, settings)

        results = await service.retrieve(db_session, "query", owner.id)

        assert results[0].content == "reranker favorite"


class TestIntelligentRanking:
    async def test_recency_boost_can_reorder_equally_relevant_chunks(self, db_session):
        owner = await make_user(db_session)
        old_doc = await make_document(db_session, owner_id=owner.id, filename="old.pdf")
        new_doc = await make_document(db_session, owner_id=owner.id, filename="new.pdf")

        await make_chunk(
            db_session, document=old_doc, content="policy text old", vector_id=1
        )
        await make_chunk(
            db_session, document=new_doc, content="policy text new", vector_id=2
        )
        await db_session.commit()

        vector_store = FakeVectorStore()
        # Identical dense similarity for both — only the recency boost should
        # decide the final order.
        same_vector = _unit_vector(0)
        await vector_store.add(
            [1, 2],
            np.stack([same_vector, same_vector]),
            metadatas=[
                {"owner_id": owner.id, "created_at": iso_days_ago(400)},
                {"owner_id": owner.id, "created_at": iso_days_ago(0)},
            ],
        )

        embeddings = ExactMapEmbeddingService({"policy text": same_vector})
        settings = _settings(
            enterprise_intelligent_ranking_enabled=True,
            enterprise_recency_boost_weight=0.5,
            enterprise_recency_half_life_days=30,
            enterprise_structural_context_boost_weight=0.0,
        )
        service = EnterpriseRetrievalService(embeddings, vector_store, settings)

        results = await service.retrieve(db_session, "policy text", owner.id)

        assert results[0].content == "policy text new"


class TestMultiQueryDisabledByDefault:
    async def test_does_not_require_an_llm_when_disabled(self, db_session):
        # If multi-query were accidentally enabled here, constructing the service
        # would try to import langchain_openai / build a ChatOpenAI client
        # and fail without OPENAI_API_KEY — this test documents that the
        # default configuration never does that.
        owner = await make_user(db_session)
        doc = await make_document(db_session, owner_id=owner.id)
        await make_chunk(
            db_session, document=doc, content="overdraft policy", vector_id=1
        )
        await db_session.commit()

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        vector = await embeddings.embed_query("overdraft policy")
        await vector_store.add(
            [1], np.stack([vector]), metadatas=[{"owner_id": owner.id}]
        )

        settings = _settings()
        assert settings.enterprise_multi_query_enabled is False
        service = EnterpriseRetrievalService(embeddings, vector_store, settings)

        results = await service.retrieve(db_session, "overdraft policy", owner.id)
        assert len(results) == 1
