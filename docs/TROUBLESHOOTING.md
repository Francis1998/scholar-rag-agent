# Troubleshooting

## Install Fails On Unsupported Python

The project requires Python 3.11+. Install a supported interpreter and rerun:

```bash
uv sync --extra dev
```

## API Starts But Returns No Evidence

Ingest at least one document before querying:

```bash
uv run python scripts/demo_local.py
```

For API usage, call `/ingest/text` before `/query`.

## Live Providers Do Not Respond

Live LLM providers are optional. If provider keys are missing, tests and demos use
the deterministic fake adapter. Set provider keys in `.env` only when live calls
are required.

## GraphRAG Finds Few Entities

Install the optional NLP stack and spaCy model for stronger NER:

```bash
uv sync --extra dev --extra all
uv run python -m spacy download en_core_web_sm
```

Without spaCy, the fallback extractor still supports deterministic tests and demos.

## CI Fails On Formatting

Run the same commands locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest tests/ -v
```
