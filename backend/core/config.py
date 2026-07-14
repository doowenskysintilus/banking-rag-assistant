"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Banking RAG Assistant"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = False

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:3000"]

    # --- Security ---
    secret_key: str = Field(..., min_length=16)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Urlsafe-base64, 32-byte (AES-256) key for backend.security.encryption —
    # see FieldEncryptor.generate_key(). Left blank, a key is derived from
    # `secret_key` instead so encryption works out of the box in dev/test; a
    # real deployment should set its own independent value (see the docstring
    # on backend.security.encryption._derive_key_from_secret for why).
    encryption_key: str = ""

    # Brute-force protection (backend.services.auth_service.AuthService).
    max_failed_login_attempts: int = 5
    account_lockout_minutes: int = 15

    # OAuth2 (backend.api.v1.routers.auth's /auth/token) — the tokenUrl FastAPI's
    # OAuth2PasswordBearer scheme advertises to Swagger's "Authorize" button.
    oauth2_token_url: str = "api/v1/auth/token"

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./banking_rag.db"

    # --- OpenAI chat (backend.llm.client) — `openai_api_key` itself lives
    # down in the Embeddings section below and is shared with
    # backend.retrieval.embeddings' OpenAI embedding provider (one API key,
    # one vendor, for both).
    openai_chat_model: str = "gpt-4o"
    openai_chat_max_tokens: int = 2048
    openai_chat_temperature: float = 0.3

    # --- Vector store (backend.retrieval.vector_store) ---
    vector_store_provider: Literal["faiss", "pinecone"] = "faiss"
    faiss_index_dir: str = "./vector_db/faiss_indices"
    # Dimension the FAISS index was built for. Must match the active embedding
    # provider's output dimension — backend.retrieval.embeddings.EmbeddingService
    # validates this at startup and raises a clear error on mismatch (e.g. after
    # switching EMBEDDING_PROVIDER without reindexing).
    embedding_dim: int = 1024
    # FAISS `index_factory` spec: "Flat" (exact, no training, default — fine up to
    # low millions of vectors), "HNSW32" (approximate, no training, faster at scale),
    # "IVF100,Flat" (approximate, REQUIRES enough training vectors on first add —
    # only use this once you know your corpus size well enough to size `nlist`).
    faiss_index_factory: str = "Flat"
    # How many extra candidates to over-fetch from FAISS when a metadata filter is
    # applied (post-filtering can otherwise return fewer than top_k results).
    faiss_filter_overfetch_factor: int = 5
    # Safety cap on the over-fetch retry loop (doubles fetch_k each attempt) for a
    # highly selective filter in a large index.
    faiss_max_filter_retries: int = 5
    # RRF smoothing constant for hybrid_search's dense+BM25 fusion — 60 is the
    # standard default from the original Reciprocal Rank Fusion paper.
    faiss_hybrid_rrf_k: int = 60

    pinecone_api_key: str = ""
    pinecone_environment: str = (
        ""  # legacy pod-based deployments only; unused by ServerlessSpec below
    )
    pinecone_index_name: str = "banking-rag"
    # Used to create the index (via ServerlessSpec) if it doesn't already exist.
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: Literal["cosine", "dotproduct", "euclidean"] = "cosine"
    pinecone_create_index_if_missing: bool = True
    pinecone_namespace: str = ""
    # Hybrid search on Pinecone is done the same way as on FAISS — a local BM25
    # index fused with Pinecone's native dense query via RRF — since Pinecone's own
    # sparse-vector hybrid mode needs a differently-configured index. That local
    # BM25 index is rebuilt at startup by paging through the remote index (bounded
    # by this limit); very large indexes should prefer FAISS if lexical recall
    # over the *entire* corpus matters, since this rebuild is necessarily partial.
    pinecone_bm25_rebuild_limit: int = 10_000

    # --- Embeddings (backend.retrieval.embeddings) ---
    # "auto" tries providers in priority order — openai > bge_large > instructor_xl —
    # based on what's configured/installed (see backend.retrieval.embeddings.selector).
    # Pin to a specific provider to disable that fallback behavior.
    embedding_provider: Literal["auto", "openai", "bge_large", "instructor_xl"] = "auto"

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dim: int = 3072

    bge_model_name: str = "BAAI/bge-large-en-v1.5"
    bge_embedding_dim: int = 1024

    instructor_model_name: str = "hkunlp/instructor-xl"
    instructor_embedding_dim: int = 768
    # Instruction pair used by Instructor-XL, which embeds asymmetric
    # (instruction, text) pairs rather than raw text.
    instructor_document_instruction: str = (
        "Represent the financial document for retrieval:"
    )
    instructor_query_instruction: str = (
        "Represent the question for retrieving supporting financial documents:"
    )

    embedding_batch_size: int = 32
    embedding_max_concurrency: int = 4

    embedding_cache_backend: Literal["redis", "memory", "none"] = "redis"
    embedding_cache_ttl_seconds: int = 2_592_000  # 30 days
    redis_url: str = "redis://localhost:6379/0"

    # --- RAG ---
    rag_top_k: int = 5
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 100
    # "hybrid" combines dense (embedding) and lexical (BM25) search via reciprocal
    # rank fusion — better recall on exact terms (account/clause numbers, defined
    # terms) that embeddings alone can miss. "vector" is dense-only. Only used when
    # rag_retrieval_engine="native" (see below).
    rag_retrieval_mode: Literal["vector", "hybrid"] = "hybrid"
    # "native" = backend.retrieval.vector_store.VectorStoreBase.search/hybrid_search
    # directly (no LangChain, lighter weight). "enterprise" = the LangChain-based
    # pipeline in backend.retrieval.enterprise (ensemble hybrid fusion, optional
    # multi-query expansion, cross-encoder reranking, intelligent metadata ranking).
    rag_retrieval_engine: Literal["native", "enterprise"] = "enterprise"

    # --- Enterprise retrieval (backend.retrieval.enterprise) ---
    # Candidates each base retriever fetches before ensemble fusion/reranking.
    enterprise_dense_top_k: int = 20
    enterprise_bm25_top_k: int = 20
    # [dense_weight, bm25_weight] for the LangChain EnsembleRetriever's weighted RRF.
    enterprise_ensemble_weights: tuple[float, float] = (0.5, 0.5)
    # Multi-query expansion asks the LLM to rephrase the question N ways and unions
    # the retrieved results — better recall on ambiguous questions, at the cost of
    # one extra LLM call per retrieval. Opt-in since it adds real latency/cost.
    enterprise_multi_query_enabled: bool = False
    enterprise_multi_query_count: int = 3
    # Reranking (backend.retrieval.reranking) re-scores the fused candidates with a
    # slower but far more precise (query, document) joint model — cheap enough
    # (local, small model) to leave on by default. "auto" prefers BGE (stronger,
    # multilingual) and falls back to the generic cross-encoder; pin explicitly to
    # disable that fallback.
    enterprise_reranking_enabled: bool = True
    enterprise_reranker_provider: Literal["auto", "bge", "cross_encoder"] = "auto"
    bge_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    cross_encoder_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enterprise_rerank_top_n: int = 5
    # The concrete "reduce hallucinations" lever: candidates scoring below this
    # calibrated relevance threshold are dropped entirely rather than merely
    # ranked lower — see backend.retrieval.reranking.service.RerankingService.
    # If *nothing* clears it, no context reaches the LLM at all (it says "I don't
    # have enough information" per backend.llm.prompts.SYSTEM_PROMPT) instead of
    # answering from noise.
    enterprise_reranker_min_relevance_score: float = 0.1
    # Final business-rule-aware adjustment on top of the reranked scores (recency,
    # structural-context boosts) — see backend.retrieval.enterprise.intelligent_ranker.
    enterprise_intelligent_ranking_enabled: bool = True
    enterprise_recency_boost_weight: float = 0.05
    enterprise_recency_half_life_days: float = 180.0
    enterprise_structural_context_boost_weight: float = 0.02
    # The per-owner BM25 corpus (built from that owner's current document chunks)
    # is cached for this long before being rebuilt from the database.
    enterprise_bm25_cache_ttl_seconds: int = 300

    # --- Chunking (backend.ingestion.chunking) ---
    # "fixed" = legacy character-window baseline; "recursive" = structure-aware,
    # size-bounded, no embeddings needed; "semantic"/"hybrid" additionally refine
    # section splits by embedding-similarity breakpoints (hybrid only does so for
    # sections long enough to make it worthwhile — see chunking_semantic_min_units).
    chunking_strategy: Literal["fixed", "recursive", "semantic", "hybrid"] = "hybrid"
    chunking_max_chars: int = 1000
    chunking_min_chars: int = 200
    chunking_overlap_sentences: int = 2
    chunking_semantic_breakpoint_percentile: float = 95.0
    chunking_semantic_min_units: int = 6

    # --- Uploads ---
    max_upload_size_mb: int = 20
    document_storage_dir: str = "./storage/documents"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — import this instead of instantiating Settings()."""
    return Settings()
