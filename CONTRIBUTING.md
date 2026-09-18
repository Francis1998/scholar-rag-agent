# Contributing to scholar-rag-agent

Thank you for considering a contribution!

## Development Setup

```bash
git clone https://github.com/Francis1998/scholar-rag-agent.git
cd scholar-rag-agent
uv sync --extra dev
pre-commit install
```

## Running Tests

```bash
uv run pytest tests/ -v --tb=short
```

### Wheel Packaging

```bash
uv run pytest tests/test_packaging.py -v
```

These tests build a real wheel using Hatchling from the dev extras, without
downloading build tools. Runtime imports run in a temporary directory with
`python -I -S`, using only the wheel and explicit dependency paths; repository
imports, `PYTHONPATH`, and editable-install `.pth` hooks cannot hide missing
wheel modules. The tests check `config`, its API/router consumers, and all three
console entry points. Ingest and evaluation run locally; the API server is mocked
so the check neither calls a provider nor starts a long-running server.

## Coding Standards

- Python 3.11+
- Type annotations on all functions
- Google-style docstrings
- Ruff for linting and formatting (`uv run ruff check . && uv run ruff format .`)

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Ensure tests pass and coverage stays at or above 70%
3. Update relevant documentation
4. Open a PR with a clear description of the change

Do not push bulk documentation or workflow rewrites directly to `main`. Use a branch and PR so CI and review can catch template drift.

## Commit Message Format

```
<type>(<scope>): <short summary>

type: feat | fix | docs | refactor | test | chore
```

*Updated: 2026-04-06*
