# OpenAlex Sources (Venues) Source Guide

![OpenAlex sources discovery flow](../assets/openalex_sources_source.gif)

Use this guide when wiring OpenAlex **sources/venues** into **scholar-rag-agent**.
Discovery is deterministic HTTP against the public OpenAlex sources API —
enrichment with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 is optional
after documents are collected.

## Why OpenAlex sources

OpenAlex indexes journals, repositories, and conferences with ISSNs, host
organizations, OA flags, and bibliometrics. Venue-aware retrieval complements
paper-first connectors (works, authors, concepts, institutions).

## Usage

```python
from ingestion.openalex_sources import OpenAlexSourcesConnector

docs = await OpenAlexSourcesConnector(mailto="dev@example.org").search(
    "Nature",
    max_results=5,
)
```

Bare OpenAlex source ids such as `S137773608` resolve directly.

## What you get

| Field | Source |
|---|---|
| `title` | `display_name` |
| `text` | Venue descriptor (type, host, ISSN, OA, works, citations, h-index) |
| `metadata.source_type` | `openalex_sources` |
| `metadata.openalex_source_id` | Bare `S####` id |
| `metadata.issn_l` / `issn` | ISSN-L and ISSN list |
| `metadata.is_oa` | Open-access venue flag |

## Safety notes

- Public OpenAlex API only — no authenticated endpoints.
- Failures return an empty list rather than raising.
- `max_results` caps response size.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `openalex`, `arxiv`, `pubmed`, `llm`, `python`
