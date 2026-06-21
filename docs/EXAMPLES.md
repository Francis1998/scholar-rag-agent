# Usage Examples

## Run The Local Demo

```bash
uv run python scripts/demo_local.py
```

This ingests a fixture paper, runs the Observe -> Decide -> Act agent, prints the
planner trace, and returns a cited answer.

## Start The API

```bash
uv run uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

## Ingest Text Through The API

```bash
curl -X POST http://127.0.0.1:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "title": "GraphRAG API Fixture",
    "text": "GraphRAG connects entities for multi-hop scientific retrieval.",
    "source": "fixture"
  }'
```

## Query The Agent

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does GraphRAG connect?"}'
```

## Evaluate Retrieval

```bash
uv run python scripts/evaluate_retrieval.py
```
