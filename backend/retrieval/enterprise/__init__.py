"""Enterprise retrieval: a LangChain-based retrieval pipeline layered on top of
`backend.retrieval.vector_store` (FAISS/Pinecone) and `backend.retrieval.embeddings`.

Composes, per query: dense search + BM25 fused via LangChain's `EnsembleRetriever`
(hybrid search / score fusion) → optional LLM-driven multi-query expansion →
optional cross-encoder reranking → a final business-rule-aware ranking pass.
See `backend.retrieval.enterprise.service.EnterpriseRetrievalService`.
"""

from backend.retrieval.enterprise.service import (
    EnterpriseRetrievalService,
    RetrievalResult,
)

__all__ = ["EnterpriseRetrievalService", "RetrievalResult"]
