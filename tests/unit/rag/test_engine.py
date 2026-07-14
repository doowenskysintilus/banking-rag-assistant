import numpy as np
import pytest
from sqlalchemy import select

from backend.core.config import Settings
from backend.models.query_log import RagQueryLog
from backend.rag.engine import DocumentUsed, RAGEngine
from backend.retrieval.enterprise.service import RetrievalResult
from backend.retrieval.reranking.service import RerankingService
from tests.unit.rag.conftest import (
    FakeEmbeddingService,
    FakeVectorStore,
    make_chunk,
    make_document,
    make_user,
)
from tests.unit.rag.fakes import FakeOpenAIClient, FakeRerankerProvider

DIM = 4


def _settings(**overrides) -> Settings:
    defaults = {
        "secret_key": "x" * 16,
        "rag_retrieval_engine": "native",
        "rag_retrieval_mode": "hybrid",
        "rag_top_k": 5,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class FakeEnterpriseRetrieval:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, str, int | None]] = []

    async def retrieve(
        self, db, query, owner_id, filters=None, top_k=None
    ) -> list[RetrievalResult]:
        self.calls.append((query, owner_id, top_k))
        return self._results


async def _seed_owner_with_chunk(
    db_session,
    owner_id: str,
    content: str,
    filename: str = "policy.pdf",
    header_path: str = "",
):
    doc = await make_document(db_session, owner_id=owner_id, filename=filename)
    chunk = await make_chunk(
        db_session, document=doc, content=content, vector_id=1, header_path=header_path
    )
    await db_session.commit()
    return doc, chunk


async def _add_matching_vector(
    vector_store: FakeVectorStore,
    embeddings: FakeEmbeddingService,
    owner_id: str,
    content: str,
):
    vector = await embeddings.embed_query(content)
    await vector_store.add([1], np.stack([vector]), metadatas=[{"owner_id": owner_id}])


class TestConstructorValidation:
    def test_enterprise_engine_requires_enterprise_retrieval_instance(self):
        with pytest.raises(ValueError):
            RAGEngine(
                embedding_service=FakeEmbeddingService(dim=DIM),
                vector_store=FakeVectorStore(),
                reranking_service=RerankingService(FakeRerankerProvider()),
                openai_client=FakeOpenAIClient(),
                settings=_settings(rag_retrieval_engine="enterprise"),
                enterprise_retrieval=None,
            )


class TestNativeEnginePipeline:
    async def test_full_pipeline_returns_grounded_answer_with_citations(
        self, db_session
    ):
        owner = await make_user(db_session)
        doc, chunk = await _seed_owner_with_chunk(
            db_session,
            owner.id,
            "Overdraft fee is $35 per item.",
            filename="overdraft_policy.pdf",
            header_path="Article 5",
        )

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        await _add_matching_vector(
            vector_store, embeddings, owner.id, "Overdraft fee is $35 per item."
        )

        reranking = RerankingService(
            FakeRerankerProvider(default_score=0.9), top_n=5, min_relevance_score=0.1
        )
        llm = FakeOpenAIClient(
            answer_text="The overdraft fee is $35.", input_tokens=120, output_tokens=15
        )

        engine = RAGEngine(embeddings, vector_store, reranking, llm, _settings())
        response = await engine.answer(
            db_session, "What is the overdraft fee?", owner.id, conversation_id=None
        )

        assert response.answer == "The overdraft fee is $35."
        assert len(response.sources) == 1
        assert response.sources[0].document_filename == "overdraft_policy.pdf"
        assert response.sources[0].header_path == "Article 5"
        assert response.sources[0].chunk_id == chunk.id
        assert response.documents_used == [
            DocumentUsed(
                document_id=doc.id, filename="overdraft_policy.pdf", chunk_count=1
            )
        ]
        assert response.confidence.level in ("high", "medium")
        assert response.input_tokens == 120
        assert response.output_tokens == 15

        for key in (
            "retrieval_ms",
            "reranking_ms",
            "context_builder_ms",
            "prompt_builder_ms",
            "llm_ms",
        ):
            assert key in response.stage_durations_ms

        # The model actually saw the source marker with the real filename.
        assert "overdraft_policy.pdf" in llm.calls[0][-1]["content"]

        # And the query was durably audited.
        result = await db_session.execute(
            select(RagQueryLog).where(RagQueryLog.user_id == owner.id)
        )
        entry = result.scalars().one()
        assert entry.confidence_level == response.confidence.level
        assert entry.input_tokens == 120
        assert entry.num_context_chunks == 1
        assert entry.error is None

    async def test_never_returns_another_owners_chunks(self, db_session):
        owner = await make_user(db_session)
        attacker = await make_user(db_session)
        await _seed_owner_with_chunk(
            db_session, owner.id, "Victim's confidential policy."
        )

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        await _add_matching_vector(
            vector_store, embeddings, owner.id, "Victim's confidential policy."
        )

        reranking = RerankingService(
            FakeRerankerProvider(), top_n=5, min_relevance_score=0.1
        )
        llm = FakeOpenAIClient(answer_text="I don't have enough information.")

        engine = RAGEngine(embeddings, vector_store, reranking, llm, _settings())
        response = await engine.answer(
            db_session, "Victim's confidential policy.", attacker.id
        )

        assert response.sources == []
        assert response.confidence.level == "low"
        assert (
            "No relevant internal documents were found"
            in llm.calls[0][-1]["content"]
        )

    async def test_low_relevance_candidates_are_dropped_by_the_confidence_gate(
        self, db_session
    ):
        owner = await make_user(db_session)
        await _seed_owner_with_chunk(
            db_session, owner.id, "Some tangential policy text."
        )

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        await _add_matching_vector(
            vector_store, embeddings, owner.id, "Some tangential policy text."
        )

        # Reranker never clears a very high bar -> everything gets filtered out.
        reranking = RerankingService(
            FakeRerankerProvider(default_score=0.2), top_n=5, min_relevance_score=0.9
        )
        llm = FakeOpenAIClient()

        engine = RAGEngine(embeddings, vector_store, reranking, llm, _settings())
        response = await engine.answer(db_session, "question", owner.id)

        assert response.sources == []
        assert response.confidence.level == "low"


class TestEnterpriseEngineDelegation:
    async def test_delegates_retrieval_and_reranking_as_a_single_stage(
        self, db_session
    ):
        owner = await make_user(db_session)
        doc = await make_document(
            db_session, owner_id=owner.id, filename="regulation.pdf"
        )
        await db_session.commit()

        fake_enterprise = FakeEnterpriseRetrieval(
            [
                RetrievalResult(
                    document_id=doc.id,
                    chunk_id="c1",
                    content="Capital ratio is 4.5%.",
                    score=0.95,
                    header_path="Section 3",
                )
            ]
        )
        embeddings = FakeEmbeddingService(dim=DIM)
        vector_store = FakeVectorStore()
        reranking = RerankingService(
            FakeRerankerProvider(), top_n=5, min_relevance_score=0.1
        )
        llm = FakeOpenAIClient(answer_text="4.5%.")

        settings = _settings(rag_retrieval_engine="enterprise", rag_top_k=5)
        engine = RAGEngine(
            embeddings,
            vector_store,
            reranking,
            llm,
            settings,
            enterprise_retrieval=fake_enterprise,
        )

        response = await engine.answer(
            db_session, "What is the capital ratio?", owner.id
        )

        assert response.answer == "4.5%."
        assert response.sources[0].document_filename == "regulation.pdf"
        assert response.sources[0].header_path == "Section 3"
        # Enterprise mode already reranked internally — no separate stage here.
        assert "reranking_ms" not in response.stage_durations_ms
        assert fake_enterprise.calls[0] == ("What is the capital ratio?", owner.id, 5)

    async def test_empty_enterprise_result_is_treated_as_low_confidence(
        self, db_session
    ):
        owner = await make_user(db_session)
        fake_enterprise = FakeEnterpriseRetrieval([])
        settings = _settings(rag_retrieval_engine="enterprise")
        engine = RAGEngine(
            FakeEmbeddingService(dim=DIM),
            FakeVectorStore(),
            RerankingService(FakeRerankerProvider()),
            FakeOpenAIClient(),
            settings,
            enterprise_retrieval=fake_enterprise,
        )

        response = await engine.answer(db_session, "question", owner.id)
        assert response.confidence.level == "low"


class TestFailureAuditing:
    async def test_llm_failure_is_still_logged_with_the_error(self, db_session):
        owner = await make_user(db_session)
        await _seed_owner_with_chunk(db_session, owner.id, "Some policy text.")

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        await _add_matching_vector(
            vector_store, embeddings, owner.id, "Some policy text."
        )

        reranking = RerankingService(
            FakeRerankerProvider(), top_n=5, min_relevance_score=0.1
        )
        llm = FakeOpenAIClient(raise_error=RuntimeError("OpenAI API is down"))

        engine = RAGEngine(embeddings, vector_store, reranking, llm, _settings())

        with pytest.raises(RuntimeError):
            await engine.answer(db_session, "question", owner.id)

        result = await db_session.execute(
            select(RagQueryLog).where(RagQueryLog.user_id == owner.id)
        )
        entry = result.scalars().one()
        assert "OpenAI API is down" in entry.error


class TestStreaming:
    async def test_prepare_stage_completes_before_the_stream_is_consumed(
        self, db_session
    ):
        owner = await make_user(db_session)
        await _seed_owner_with_chunk(db_session, owner.id, "Some policy text.")

        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=DIM)
        await _add_matching_vector(
            vector_store, embeddings, owner.id, "Some policy text."
        )

        reranking = RerankingService(
            FakeRerankerProvider(), top_n=5, min_relevance_score=0.1
        )
        llm = FakeOpenAIClient(stream_chunks=["Hel", "lo"], answer_text="Hello")

        engine = RAGEngine(embeddings, vector_store, reranking, llm, _settings())
        prepared, stream = await engine.answer_stream(db_session, "question", owner.id)

        assert len(prepared.sources) == 1

        collected_text = ""
        final = None
        async for chunk in stream:
            if chunk.is_final:
                final = chunk.final_response
            else:
                collected_text += chunk.text

        assert collected_text == "Hello"
        assert final.text == "Hello"

        await engine.log_query(
            db_session,
            user_id=owner.id,
            conversation_id=None,
            question="question",
            prepared=prepared,
            llm_response=final,
            total_ms=10.0,
        )
        result = await db_session.execute(
            select(RagQueryLog).where(RagQueryLog.user_id == owner.id)
        )
        assert result.scalars().one().input_tokens == final.input_tokens
