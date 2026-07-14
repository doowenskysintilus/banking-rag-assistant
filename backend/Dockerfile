# syntax=docker/dockerfile:1
#
# Multi-stage build for the FastAPI backend. This image ships the FULL
# requirements.txt — including the optional local-model ML stack
# (sentence-transformers, FlagEmbedding, InstructorEmbedding) so every
# embedding/reranker provider in backend.retrieval is actually available at
# runtime, not just the API-based ones. That makes this a large image
# (multi-GB, mostly PyTorch); if a given deployment only ever uses
# EMBEDDING_PROVIDER=openai and ENTERPRISE_RERANKER_PROVIDER=cross_encoder,
# trimming sentence-transformers/FlagEmbedding/InstructorEmbedding from
# requirements.txt before building will shrink this considerably — the
# selector modules (backend.retrieval.*.selector) already degrade gracefully
# when a provider's package isn't installed.
#
# Build context is the repo root (see docker-compose.yml / CI):
#   docker build -f backend/Dockerfile -t banking-rag-backend .

ARG PYTHON_VERSION=3.14-slim-bookworm

# --- Stage 1: build a wheelhouse ---------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

WORKDIR /build

# build-essential is only needed to compile a handful of transitive deps that
# don't ship prebuilt wheels for every platform — it never reaches the final
# image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# --- Stage 2: runtime ---------------------------------------------------------
FROM python:${PYTHON_VERSION} AS runtime

# Kept in the runtime image (not just builder) because a couple of packages
# (faiss-cpu's OpenMP runtime, some tokenizer wheels) dynamically link against
# libgomp1 at import time, not just build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

COPY --from=builder /build/wheels /wheels
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

COPY backend ./backend
COPY alembic.ini ./alembic.ini
COPY migrations ./migrations
COPY backend/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Writable directories the app needs at runtime (uploaded documents, the FAISS
# index, and the SQLite dev DB if DATABASE_URL isn't overridden) — mount these
# as volumes in docker-compose/k8s so they survive container restarts.
RUN mkdir -p /app/storage/documents /app/vector_db/faiss_indices \
    && chown -R app:app /app

USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/app/.local/bin:${PATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]

# One worker per container: horizontal scaling is done via replicas
# (docker-compose `deploy.replicas` / Kubernetes HPA — see k8s/base), not
# multiple uvicorn workers sharing one container's resource limits.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
