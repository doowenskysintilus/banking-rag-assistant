.PHONY: help up down build logs ps migrate test test-backend test-frontend lint format k8s-diff k8s-apply

help:
	@echo "Docker Compose:"
	@echo "  make build         Build all images"
	@echo "  make up            Start the full stack (detached)"
	@echo "  make down          Stop the stack (keeps volumes)"
	@echo "  make logs          Follow logs for all services"
	@echo "  make ps            Show service status"
	@echo "  make migrate       Apply Alembic migrations against the compose Postgres"
	@echo ""
	@echo "Testing / linting (run against the local Python/Node toolchain, not Docker):"
	@echo "  make test          Run the full backend test suite (two isolated invocations — see ci.yml)"
	@echo "  make test-frontend Typecheck + lint + build the frontend"
	@echo "  make lint          ruff check + format --check"
	@echo "  make format        ruff format (writes changes)"
	@echo ""
	@echo "Kubernetes (kustomize):"
	@echo "  make k8s-diff      Render the production overlay to stdout (no cluster access needed)"
	@echo "  make k8s-apply     kubectl apply -k the production overlay (needs a configured kubeconfig)"

# --- Docker Compose ------------------------------------------------------------

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

migrate:
	docker compose run --rm migrate

# --- Backend testing / linting --------------------------------------------------

test:
	SECRET_KEY=$${SECRET_KEY:-local-dev-secret-key-change-me} pytest tests/unit --ignore=tests/unit/evaluation -q
	SECRET_KEY=$${SECRET_KEY:-local-dev-secret-key-change-me} pytest tests/unit/evaluation -q

lint:
	ruff check backend migrations tests
	ruff format --check backend migrations tests

format:
	ruff format backend migrations tests

# --- Frontend --------------------------------------------------------------------

test-frontend:
	cd frontend && npx tsc -b --noEmit && npx eslint . && npm run build

# --- Kubernetes ------------------------------------------------------------------

k8s-diff:
	kustomize build k8s/overlays/production

k8s-apply:
	kustomize build k8s/overlays/production | kubectl apply -f -
