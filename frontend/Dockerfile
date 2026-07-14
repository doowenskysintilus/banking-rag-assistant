# syntax=docker/dockerfile:1
#
# Multi-stage build: compile the Vite/React SPA, then serve the static output
# with a minimal nginx. Build context is frontend/ itself (see
# docker-compose.yml / CI — unlike the backend, whose context is the repo root).
#
#   docker build -t banking-rag-frontend ./frontend

# --- Stage 1: build ------------------------------------------------------------
FROM node:22-alpine AS builder

WORKDIR /app

# VITE_API_BASE_URL is a *build-time* value (Vite inlines import.meta.env at
# build time, there is no runtime env injection here) — left empty by default
# so the app calls relative `/api/v1/...` paths, which only works when this
# image is served from behind the same-origin reverse proxy in
# nginx/ (docker-compose) or the Ingress (k8s/base) that also routes to the
# backend. Only set this to an absolute URL for a genuinely cross-origin
# deployment, and make sure the backend's CORS_ORIGINS allows it.
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- Stage 2: runtime -----------------------------------------------------------
FROM nginx:1.31-alpine AS runtime

RUN rm -rf /usr/share/nginx/html/*
COPY --from=builder /app/dist /usr/share/nginx/html
COPY frontend.nginx.conf /etc/nginx/conf.d/default.conf

# nginx:alpine's image already runs as an unprivileged user via
# nginx-unprivileged conventions in newer tags, but this base still starts as
# root to bind :80 then drop privileges internally — no extra USER directive
# needed here; see nginx/Dockerfile for the edge proxy's equivalent notes.

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
