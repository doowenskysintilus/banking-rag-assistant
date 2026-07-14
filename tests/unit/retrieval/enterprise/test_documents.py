from backend.retrieval.enterprise.documents import (
    fetch_owner_documents,
    hydrate_search_results,
)
from backend.retrieval.vector_store.types import SearchResult
from tests.unit.retrieval.enterprise.conftest import (
    make_chunk,
    make_document,
    make_user,
)


async def test_fetch_owner_documents_returns_only_that_owners_current_chunks(
    db_session,
):
    owner = await make_user(db_session)
    other_owner = await make_user(db_session)

    doc = await make_document(db_session, owner_id=owner.id)
    await make_chunk(
        db_session, document=doc, content="Overdraft policy text.", vector_id=1
    )

    other_doc = await make_document(db_session, owner_id=other_owner.id)
    await make_chunk(
        db_session, document=other_doc, content="Someone else's document.", vector_id=2
    )
    await db_session.commit()

    documents = await fetch_owner_documents(db_session, owner.id)

    assert len(documents) == 1
    assert documents[0].page_content == "Overdraft policy text."
    assert documents[0].metadata["vector_id"] == 1


async def test_fetch_owner_documents_excludes_superseded_versions(db_session):
    owner = await make_user(db_session)
    doc = await make_document(db_session, owner_id=owner.id, is_current=False)
    await make_chunk(db_session, document=doc, content="Old version text.", vector_id=1)
    await db_session.commit()

    documents = await fetch_owner_documents(db_session, owner.id)
    assert documents == []


async def test_fetch_owner_documents_empty_for_unknown_owner(db_session):
    assert await fetch_owner_documents(db_session, "no-such-owner") == []


async def test_hydrate_search_results_joins_real_content_and_metadata(db_session):
    owner = await make_user(db_session)
    doc = await make_document(db_session, owner_id=owner.id)
    await make_chunk(
        db_session,
        document=doc,
        content="Overdraft policy text.",
        vector_id=1,
        header_path="Article 5",
    )
    await db_session.commit()

    hits = [
        SearchResult(
            vector_id=1, score=0.9, metadata={"owner_id": owner.id, "language": "en"}
        )
    ]
    documents = await hydrate_search_results(db_session, hits, owner.id)

    assert len(documents) == 1
    result = documents[0]
    assert result.page_content == "Overdraft policy text."
    assert result.metadata["score"] == 0.9
    assert result.metadata["header_path"] == "Article 5"
    assert result.metadata["language"] == "en"
    assert result.metadata["document_id"] == doc.id


async def test_hydrate_search_results_skips_hits_belonging_to_another_owner(db_session):
    owner = await make_user(db_session)
    attacker = await make_user(db_session)

    victim_doc = await make_document(db_session, owner_id=owner.id)
    await make_chunk(
        db_session, document=victim_doc, content="Victim's private data.", vector_id=1
    )
    await db_session.commit()

    # Attacker's query somehow gets a hit for the victim's vector_id (e.g. a stale/
    # tampered vector-store metadata entry) — the SQL join must not honor it.
    hits = [SearchResult(vector_id=1, score=0.99, metadata={"owner_id": attacker.id})]
    documents = await hydrate_search_results(db_session, hits, attacker.id)

    assert documents == []


async def test_hydrate_search_results_skips_stale_vector_ids(db_session):
    owner = await make_user(db_session)
    hits = [SearchResult(vector_id=999, score=0.5, metadata={})]
    documents = await hydrate_search_results(db_session, hits, owner.id)
    assert documents == []


async def test_hydrate_search_results_empty_hits_returns_empty(db_session):
    assert await hydrate_search_results(db_session, [], "owner-1") == []
