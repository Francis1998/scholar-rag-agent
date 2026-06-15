.PHONY: help install test lint format clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run test suite
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:  ## Run ruff linter
	ruff check .

format:  ## Auto-format with ruff
	ruff format .
	ruff check --fix .

clean:  ## Remove build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info

# Generated: 2026-06-15
