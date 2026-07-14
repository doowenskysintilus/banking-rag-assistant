"""Vector store: a Strategy-pattern abstraction over FAISS (local) and Pinecone
(managed cloud), with metadata filtering and dense+lexical hybrid search on both.

`VectorStoreBase` is the single interface the rest of the app depends on
(`backend.rag.engine`, `backend.workers.tasks`, `backend.services.document_service`
never import a concrete implementation directly) — swapping providers is a config
change (`VECTOR_STORE_PROVIDER=faiss|pinecone`), not a code change. See
`backend.retrieval.vector_store.factory` for the strategy selection and
`tests/unit/retrieval/vector_store/test_contract.py` for the shared behavior suite
both implementations are held to.
"""

from backend.retrieval.vector_store.factory import build_vector_store, get_vector_store
from backend.retrieval.vector_store.faiss_store import FAISSVectorStore
from backend.retrieval.vector_store.pinecone_store import PineconeVectorStore
from backend.retrieval.vector_store.types import Filter, SearchResult, VectorStoreBase

__all__ = [
    "VectorStoreBase",
    "SearchResult",
    "Filter",
    "FAISSVectorStore",
    "PineconeVectorStore",
    "build_vector_store",
    "get_vector_store",
]
