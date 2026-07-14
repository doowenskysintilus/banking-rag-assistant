import pytest

from backend.core.config import Settings
from backend.core.exceptions import ValidationAppError
from backend.retrieval.vector_store.factory import build_vector_store
from backend.retrieval.vector_store.faiss_store import FAISSVectorStore
from backend.retrieval.vector_store.pinecone_store import PineconeVectorStore


def test_builds_faiss_store_by_default(tmp_path):
    settings = Settings(
        secret_key="x" * 16, faiss_index_dir=str(tmp_path / "faiss"), embedding_dim=8
    )
    store = build_vector_store(settings)
    assert isinstance(store, FAISSVectorStore)


def test_builds_pinecone_store_when_configured(tmp_path, monkeypatch):
    import pinecone

    from tests.unit.retrieval.vector_store.pinecone_fakes import (
        FakePineconeProject,
        FakeServerlessSpec,
    )

    project = FakePineconeProject()
    monkeypatch.setattr(pinecone, "Pinecone", project.connect)
    monkeypatch.setattr(pinecone, "ServerlessSpec", FakeServerlessSpec)

    settings = Settings(
        secret_key="x" * 16,
        vector_store_provider="pinecone",
        pinecone_api_key="pk-fake",
        embedding_dim=8,
    )
    store = build_vector_store(settings)
    assert isinstance(store, PineconeVectorStore)


def test_unknown_provider_raises_a_clear_error(tmp_path):
    settings = Settings(
        secret_key="x" * 16, faiss_index_dir=str(tmp_path / "faiss"), embedding_dim=8
    )
    settings.vector_store_provider = "qdrant"  # not registered
    with pytest.raises(ValidationAppError):
        build_vector_store(settings)


def test_build_vector_store_returns_independent_instances(tmp_path):
    settings = Settings(
        secret_key="x" * 16, faiss_index_dir=str(tmp_path / "faiss"), embedding_dim=8
    )
    first = build_vector_store(settings)
    second = build_vector_store(settings)
    assert first is not second
