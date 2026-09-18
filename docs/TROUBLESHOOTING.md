# Troubleshooting

## Install Fails On Unsupported Python

The project requires Python 3.11+. Install a supported interpreter and rerun:

```bash
uv sync --extra dev
```

## API Starts But Returns No Evidence

Ingest through `/ingest/text` before querying the API; copy the requests from the
[Quickstart](../QUICKSTART.md#4-ingest-text-and-ask-a-question).
`scripts/demo_local.py` uses a separate temporary database and does not populate
the running API.

Check that a restarted server uses the same `SCHOLAR_RAG_DATABASE_PATH` as the
ingestion run. Starting with a new temporary path creates an empty corpus.
Also inspect `result.state`, `result.error`, and grounding warnings: HTTP success
does not necessarily mean the agent completed successfully.

## Port 8000 Is Already In Use

Use another local port, for example `--host 127.0.0.1 --port 8001`, and change
the example URLs or `BASE_URL` to match. Do not stop an unrelated service to run
the demo.

## A Run Cannot Be Exported

The export endpoint returns 404 for an unknown run, 409 for a run that is
incomplete, failed, or lacks an evidence snapshot, and 422 for an unsupported
format. Old event-only runs are not silently reconstructed from today's corpus.
Inspect the [export guide](guides/EVIDENCE_EXPORT_GUIDE.md) and saved events.
If you deliberately run the question again, treat the result as a new run with
new evidence, not a repair of the historical one.

## Live Providers Do Not Respond

Live LLM providers are optional. If provider keys are missing, tests and demos use
the deterministic fake adapter. Set provider keys in `.env` only when live calls
are required. The [provider model guide](guides/PROVIDER_MODELS_GUIDE.md) explains
current model IDs and routing; `/query` uses the reasoning route, so changing the
default-provider setting alone may not select the adapter you expect.

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
