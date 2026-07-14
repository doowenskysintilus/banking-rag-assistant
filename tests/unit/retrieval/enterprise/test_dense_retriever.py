import numpy as np
import pytest

from backend.retrieval.enterprise.dense_retriever import DenseRetriever
from tests.unit.retrieval.enterprise.conftest import (
    FakeEmbeddingService,
    FakeVectorStore,
    make_chunk,
    make_document,
    make_user,
)


async def _seed(db_session, owner_id: str):
    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddingService(dim=4)

    doc = await make_document(db_session, owner_id=owner_id)
    chunk = await make_chunk(
        db_session, document=doc, content="overdraft policy details", vector_id=1
    )
    await db_session.commit()

    vector = await embeddings.embed_query("overdraft policy details")
    await vector_store.add([1], np.stack([vector]), metadatas=[{"owner_id": owner_id}])

    return vector_store, embeddings, chunk


class TestAsyncRetrieval:
    async def test_returns_hydrated_documents_matching_the_query(self, db_session):
        owner = await make_user(db_session)
        vector_store, embeddings, chunk = await _seed(db_session, owner.id)

        retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_service=embeddings,
            db=db_session,
            owner_id=owner.id,
            top_k=5,
        )

        documents = await retriever.ainvoke("overdraft policy details")

        assert len(documents) == 1
        assert documents[0].page_content == "overdraft policy details"
        assert documents[0].metadata["chunk_id"] == chunk.id

    async def test_applies_filters_to_the_underlying_search(self, db_session):
        owner = await make_user(db_session)
        vector_store, embeddings, _ = await _seed(db_session, owner.id)

        retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_service=embeddings,
            db=db_session,
            owner_id=owner.id,
            top_k=5,
            filters={"owner_id": "someone-else"},
        )

        documents = await retriever.ainvoke("overdraft policy details")
        assert documents == []

    async def test_no_matches_returns_empty_list(self, db_session):
        owner = await make_user(db_session)
        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=4)

        retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_service=embeddings,
            db=db_session,
            owner_id=owner.id,
            top_k=5,
        )
        assert await retriever.ainvoke("anything") == []

    async def test_sync_entry_point_raises_inside_a_running_event_loop(
        self, db_session
    ):
        owner = await make_user(db_session)
        vector_store = FakeVectorStore()
        embeddings = FakeEmbeddingService(dim=4)
        retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_service=embeddings,
            db=db_session,
            owner_id=owner.id,
            top_k=5,
        )
        with pytest.raises(RuntimeError):
            retriever.invoke("anything")
