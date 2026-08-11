# DataCite Event Data Source Guide

![DataCite Event Data discovery flow](../assets/datacite_events_source.gif)

Use this guide when wiring DataCite citation and usage events into
**scholar-rag-agent**. Discovery is deterministic HTTP against the public
DataCite Event Data API; enrichment with GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 is optional after documents are collected.

## Why DataCite Event Data

The DataCite DOI registry describes scholarly objects; Event Data describes
links and activity around them. Citation, reference, usage, and related-object
events let retrieval workflows surface relationships involving datasets,
software, preprints, articles, and other DOI-identified outputs.

## Usage

```python
from ingestion.datacite_events import DataCiteEventsConnector

docs = await DataCiteEventsConnector().search(
    "10.5061/dryad.qjq2bvqhq",
    max_results=5,
)
```

DOI-shaped input uses `GET /events?doi=...`; other text uses the API's general
`query` filter. `page[size]` is capped at the API maximum of 1000.

## What you get

| Field | Source |
|---|---|
| `title` | Synthesized relation plus subject/object identifiers |
| `text` | Event descriptor with relation, source, endpoints, count, and timestamp |
| `metadata.source_type` | `datacite_events` |
| `metadata.event_id` | JSON:API event resource `id` |
| `metadata.subject_id` / `object_id` | Event `subj-id` / `obj-id` |
| `metadata.subject_doi` / `object_doi` | Bare DOI when an endpoint is DOI-shaped |
| `metadata.relation_type` | `relation-type-id` |
| `metadata.event_source` | `source-id` |
| `metadata.total` | Aggregated event total |
| `metadata.occurred_at` | Event occurrence timestamp |

## Safety notes

- Public DataCite API only; no credentials are required.
- Events are relationship/attention signals, not evidence of research quality.
- Blank input and non-positive limits do not issue HTTP requests.
- Unavailable or malformed responses return an empty list.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `datacite`, `citations`, `events`, `doi`, `research-data`, `llm`, `python`
