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

## Fetch A Paper From OpenAlex

```python
import asyncio

from ingestion.openalex import OpenAlexConnector

# `mailto` is optional but routes traffic to the faster OpenAlex "polite" pool.
connector = OpenAlexConnector(mailto="you@example.org")
document = asyncio.run(connector.fetch_work("W2741809807"))
print(document.title)
print(document.text)  # abstract reconstructed from the inverted index
```

The connector normalizes OpenAlex works into the same `Document` shape as the
PDF, arXiv, and Semantic Scholar connectors, reconstructing the abstract from
OpenAlex's `abstract_inverted_index` field.

## Evaluate Retrieval

```bash
uv run python scripts/evaluate_retrieval.py
```
