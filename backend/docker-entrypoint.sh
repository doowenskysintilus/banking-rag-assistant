#!/bin/sh
# Entrypoint for the backend image. Migrations are NOT run here by default —
# with multiple replicas (the normal case in docker-compose --scale and
# always in Kubernetes), every replica's entrypoint would race to apply the
# same migration concurrently. Instead:
#   - docker-compose.yml runs a dedicated one-shot `migrate` service first
#     (`depends_on: migrate: condition: service_completed_successfully`).
#   - Kubernetes applies k8s/base/migration-job.yaml as a one-shot Job before
#     rolling out the Deployment (see the CD workflow).
#
# Set RUN_MIGRATIONS_ON_STARTUP=true only for a single-instance deployment
# (e.g. a simple VM running one container with no separate migrate step) —
# it is unsafe to set this with more than one replica running the same
# database.
set -e

if [ "${RUN_MIGRATIONS_ON_STARTUP:-false}" = "true" ]; then
    echo "RUN_MIGRATIONS_ON_STARTUP=true — applying database migrations..."
    alembic upgrade head
fi

exec "$@"
