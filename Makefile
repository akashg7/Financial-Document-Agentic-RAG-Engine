.PHONY: up down build test ingest lint clean

# Start the full stack
up:
	docker compose up --build -d

# Stop all services
down:
	docker compose down

# Rebuild without cache
build:
	docker compose build --no-cache

# Run tests inside the API container
test:
	docker compose exec api pytest tests/ -v

# Run tests locally (assumes local .env and services running)
test-local:
	pytest tests/ -v

# Ingest sample dataset
ingest:
	docker compose exec api python -m app.ingestion.loader

# Check code style
lint:
	ruff check app/ tests/

# Remove volumes and clean up
clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# View logs
logs:
	docker compose logs -f api

# Just the infrastructure (useful for local dev)
infra:
	docker compose up qdrant redis -d
