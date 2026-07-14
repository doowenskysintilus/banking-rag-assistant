# Banking RAG Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) assistant for banking and financial-services document Q&A — built with an anti-hallucination-first design, full audit traceability, and a security/compliance posture suited to regulated environments.

## Overview

Banking RAG Assistant lets internal teams (support, compliance, risk) ask natural-language questions against internal banking documents (policies, regulations, contracts) and get grounded, cited answers — never an ungrounded guess. It's built around a strict retrieval → rerank → context-build → prompt → LLM pipeline, with a confidence gate that refuses to answer when the retrieved context isn't good enough, and every query is logged for compliance audit.

## Key features

- **Anti-hallucination RAG pipeline**: hybrid (dense + BM25) retrieval, cross-encoder reranking, confidence scoring, and a strict grounding prompt that forces the model to cite sources or decline to answer.
- **Two retrieval engines**: a native hybrid pipeline and a pluggable "enterprise" engine (multi-query expansion, intelligent ranking with recency/structural boosts, ensemble fusion) selectable per deployment.
- **RAG evaluation framework**: RAGAS + DeepEval integration (faithfulness, context precision/recall, answer relevancy, hallucination rate) as an LLM-as-judge harness, wired to the same OpenAI configuration as production traffic.
- **Enterprise security**: JWT/OAuth2 authentication, role-based access control, per-owner document isolation, PII redaction, encryption at rest, and full audit logging (`backend/security`, `backend/core/audit.py`).
- **Document ingestion pipeline**: chunking, embeddings, and FAISS-backed vector storage (`backend/ingestion`, `backend/retrieval`).
- **React enterprise frontend**: chat, conversations history, document management, user administration, and an admin dashboard (`frontend/src`).
- **Production infrastructure included**: Dockerfiles, Docker Compose, Kubernetes manifests (Kustomize), Nginx reverse proxy, GitHub Actions CI/CD, and a Prometheus/Grafana/Alertmanager monitoring stack.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, SQLAlchemy (async), Alembic |
| LLM | OpenAI (Chat Completions API) |
| Retrieval | FAISS, BM25, cross-encoder reranking |
| Evaluation | RAGAS, DeepEval |
| Frontend | React 19, TypeScript, Vite, TanStack Query, Radix UI, Tailwind |
| Infra | Docker, Docker Compose, Kubernetes, Nginx, GitHub Actions |
| Observability | Prometheus, Grafana, Alertmanager |

## Project structure

```
backend/           FastAPI app: api, core, rag, retrieval, ingestion, llm, security, services, workers
frontend/          React + TypeScript SPA
k8s/                Kubernetes manifests (Kustomize base/overlays)
monitoring/         Prometheus, Grafana, Alertmanager configuration
nginx/              Reverse proxy configuration
migrations/         Alembic database migrations
tests/              Unit test suite
```

## Getting started

```bash
# Backend
cp .env.example .env   # fill in OPENAI_API_KEY, SECRET_KEY, database URL, etc.
pip install -r requirements.txt
alembic upgrade head
uvicorn backend.main:app --reload

# Frontend
cd frontend
cp .env.example .env
npm install
npm run dev
```

Or run the full stack (API, frontend, database, monitoring) with:

```bash
docker compose up --build
```

## Running tests

```bash
pytest tests/unit --ignore=tests/unit/evaluation -q
pytest tests/unit/evaluation -q
```

The evaluation suite is run as a separate process because `ragas` patches the asyncio event loop at import time (`nest_asyncio.apply()`), which is incompatible with the rest of the test suite in the same process.

## Configuration

All configuration is centralized in `backend/core/config.py` (Pydantic Settings) and documented in `.env.example`. The LLM provider is OpenAI throughout — both the production chat model and the RAG-evaluation judge model read from the same `OPENAI_API_KEY`/`OPENAI_CHAT_MODEL` settings.

## License

Proprietary — All rights reserved. See [LICENSE](LICENSE). This code is not licensed for reuse, redistribution, or modification without prior written permission.
