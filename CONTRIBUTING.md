# Contributing to scholar-rag-agent

Thank you for considering a contribution!

## Development Setup

```bash
git clone https://github.com/Francis1998/scholar-rag-agent.git
cd scholar-rag-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## Coding Standards

- Python 3.10+
- Type annotations on all functions
- Google-style docstrings
- Ruff for linting and formatting (`ruff check . && ruff format .`)

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Ensure tests pass and coverage stays above 80%
3. Update relevant documentation
4. Open a PR with a clear description of the change

## Commit Message Format

```
<type>(<scope>): <short summary>

type: feat | fix | docs | refactor | test | chore
```

*Updated: 2026-04-06*
