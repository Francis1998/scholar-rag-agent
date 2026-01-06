.PHONY: help install dev-install test lint format type-check clean docker-build

# scholar-rag-agent Makefile  —  updated 2026-01-05

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*?##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

dev-install:  ## Install with dev extras
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run test suite with coverage
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:  ## Run tests without coverage (faster)
	pytest tests/ -v --tb=short -x

lint:  ## Lint with ruff
	ruff check .

format:  ## Format with ruff
	ruff format .
	ruff check --fix .

type-check:  ## Type check with mypy
	mypy src/agent --ignore-missing-imports

clean:  ## Remove build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info

docker-build:  ## Build Docker image
	docker build -t Francis1998/scholar-rag-agent:0.8.20 .

docker-run:  ## Run Docker container
	docker run --env-file .env Francis1998/scholar-rag-agent:0.8.20

bump-patch:  ## Bump patch version
	bump2version patch
