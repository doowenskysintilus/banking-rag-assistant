# syntax=docker/dockerfile:1
#
# The edge reverse proxy — fronts both the frontend and backend services on
# one origin (see conf.d/default.conf). Build context is nginx/ itself.
#
#   docker build -t banking-rag-nginx ./nginx

FROM nginx:1.27-alpine

RUN rm /etc/nginx/conf.d/default.conf 2>/dev/null || true
COPY nginx.conf /etc/nginx/nginx.conf
COPY conf.d/ /etc/nginx/conf.d/

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:80/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
