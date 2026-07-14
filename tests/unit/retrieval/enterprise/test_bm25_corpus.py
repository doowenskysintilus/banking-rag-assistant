from backend.retrieval.enterprise import bm25_corpus
from backend.retrieval.enterprise.bm25_corpus import (
    clear_bm25_cache,
    get_owner_bm25_retriever,
    invalidate_owner_bm25_cache,
)
from tests.unit.retrieval.enterprise.conftest import (
    make_chunk,
    make_document,
    make_user,
)


async def _seed_owner_with_chunk(db_session, content: str = "overdraft policy text"):
    owner = await make_user(db_session)
    doc = await make_document(db_session, owner_id=owner.id)
    await make_chunk(db_session, document=doc, content=content, vector_id=1)
    await db_session.commit()
    return owner


def setup_function() -> None:
    clear_bm25_cache()


class TestGetOwnerBm25Retriever:
    async def test_returns_none_for_owner_with_no_chunks(self, db_session):
        owner = await make_user(db_session)
        await db_session.commit()
        assert await get_owner_bm25_retriever(db_session, owner.id, top_k=5) is None

    async def test_returns_a_working_retriever_over_the_owners_corpus(self, db_session):
        owner = await _seed_owner_with_chunk(db_session)
        retriever = await get_owner_bm25_retriever(db_session, owner.id, top_k=5)
        assert retriever is not None
        results = retriever.invoke("overdraft")
        assert len(results) == 1
        assert results[0].page_content == "overdraft policy text"

    async def test_result_is_cached_across_calls(self, db_session, monkeypatch):
        owner = await _seed_owner_with_chunk(db_session)

        first = await get_owner_bm25_retriever(db_session, owner.id, top_k=5)

        call_count = {"n": 0}
        original = bm25_corpus.fetch_owner_documents

        async def _counting_fetch(db, owner_id):
            call_count["n"] += 1
            return await original(db, owner_id)

        monkeypatch.setattr(bm25_corpus, "fetch_owner_documents", _counting_fetch)

        second = await get_owner_bm25_retriever(db_session, owner.id, top_k=5)

        assert (
            call_count["n"] == 0
        )  # cache hit — fetch_owner_documents never called again
        assert first is second

    async def test_cache_expires_after_ttl(self, db_session, monkeypatch):
        owner = await _seed_owner_with_chunk(db_session)
        clock = {"now": 1000.0}
        monkeypatch.setattr(bm25_corpus.time, "monotonic", lambda: clock["now"])

        first = await get_owner_bm25_retriever(
            db_session, owner.id, top_k=5, ttl_seconds=60
        )
        clock["now"] += 61  # past the TTL
        second = await get_owner_bm25_retriever(
            db_session, owner.id, top_k=5, ttl_seconds=60
        )

        assert first is not second

    async def test_invalidate_forces_a_rebuild_on_next_call(self, db_session):
        owner = await _seed_owner_with_chunk(db_session)

        first = await get_owner_bm25_retriever(db_session, owner.id, top_k=5)
        invalidate_owner_bm25_cache(owner.id)
        second = await get_owner_bm25_retriever(db_session, owner.id, top_k=5)

        assert first is not second

    async def test_different_owners_get_independent_caches(self, db_session):
        owner_a = await _seed_owner_with_chunk(db_session, content="overdraft policy")
        owner_b = await make_user(db_session)
        doc_b = await make_document(db_session, owner_id=owner_b.id)
        await make_chunk(
            db_session, document=doc_b, content="wire transfer limits", vector_id=2
        )
        await db_session.commit()

        retriever_a = await get_owner_bm25_retriever(db_session, owner_a.id, top_k=5)
        retriever_b = await get_owner_bm25_retriever(db_session, owner_b.id, top_k=5)

        assert retriever_a is not retriever_b
        assert retriever_a.invoke("overdraft")[0].page_content == "overdraft policy"
        assert (
            retriever_b.invoke("wire transfer")[0].page_content
            == "wire transfer limits"
        )
