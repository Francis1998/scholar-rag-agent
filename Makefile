.PHONY: help install dev-install test lint format type-check clean docker-build docker-run

# scholar-rag-agent Makefile

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*?##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	uv sync

dev-install:  ## Install with dev extras
	uv sync --extra dev
	pre-commit install

test:  ## Run test suite with coverage
	uv run pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:  ## Run tests without coverage (faster)
	uv run pytest tests/ -v --tb=short -x

lint:  ## Lint with ruff
	uv run ruff check .

format:  ## Format with ruff
	uv run ruff format .
	uv run ruff check --fix .

type-check:  ## Type check with mypy
	uv run mypy src/

clean:  ## Remove build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml dist build *.egg-info

docker-build:  ## Build Docker image
	docker build -t scholar-rag-agent:local .

docker-run:  ## Run Docker container
	docker run --rm --env-file .env -p 8000:8000 scholar-rag-agent:local
