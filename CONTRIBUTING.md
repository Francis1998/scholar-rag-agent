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

### CI Coverage Reports

CI requires both Python 3.11 and 3.12 to pass Ruff lint/format checks, mypy,
and the full test suite with at least 70% coverage. Each job uploads its
`coverage.xml` to GitHub Actions as `coverage-3.11` or `coverage-3.12`,
retained for seven days. Missing reports or failed uploads fail the job.

Download a report from the workflow run's **Artifacts** section, or use
`gh run download RUN_ID --name coverage-3.12` from a repository checkout.
Coverage delivery no longer depends on Codecov's external uploader; Codecov
dashboards are not updated automatically. No quality gate is disabled.

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

## Guide Demo GIFs

Retrieval-helper and source-connector guides embed GIFs that
`scripts/render_guide_demos.py` renders from a real run of the guide's own
`python` example. Never hand-edit or stub these images:

```bash
uv run python scripts/render_guide_demos.py --print --only bibtex-export.gif  # transcript only
uv run python scripts/render_guide_demos.py --only bibtex-export.gif          # re-render one GIF
uv run python scripts/render_guide_demos.py --check                           # all GIFs current?
```

When you change a guide example or the module behind it, re-render its GIF and
commit both. To add a demo, register a `GuideDemo` in `DEMOS`: sample input goes
in `setup`, prints in `show`, and connector demos add `Route`s that serve
payloads already recorded in `tests/test_connectors.py` (no live API calls). An
exception, any stderr output, or a request without a recorded fixture aborts the
render; fix the guide or fixture rather than the transcript.
`tests/test_demo_gifs_real.py` fails if any GIF referenced from README/docs
Markdown is missing, not a valid GIF, smaller than 200x100 px, or under 2 KB.

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

*Updated: 2026-11-07*
