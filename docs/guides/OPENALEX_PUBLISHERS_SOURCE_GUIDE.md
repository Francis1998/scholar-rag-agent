# OpenAlex Publishers Source Guide

![OpenAlex publishers discovery flow](../assets/openalex_publishers_source.gif)

Use this guide when wiring OpenAlex **publishers** into **scholar-rag-agent**.
Discovery is deterministic HTTP against the public OpenAlex publishers API —
enrichment with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 is optional
after documents are collected.

## Why OpenAlex publishers

OpenAlex indexes publishing organizations with hierarchy, country codes,
alternate titles, and bibliometrics. Publisher-aware retrieval complements
paper-first and venue (`sources`) connectors.

## Usage

```python
from ingestion.openalex_publishers import OpenAlexPublishersConnector

docs = await OpenAlexPublishersConnector(mailto="dev@example.org").search(
    "Springer",
    max_results=5,
)
```

Bare OpenAlex publisher ids such as `P4310319900` resolve directly.

## What you get

| Field | Source |
|---|---|
| `title` | `display_name` |
| `text` | Publisher descriptor (parent, countries, hierarchy, aliases, works, citations, h-index) |
| `metadata.source_type` | `openalex_publishers` |
| `metadata.openalex_publisher_id` | Bare `P####` id |
| `metadata.parent_publisher` | Parent publisher display name |
| `metadata.country_codes` | ISO country codes |

## Safety notes

- Public OpenAlex API only — no authenticated endpoints.
- Failures return an empty list rather than raising.
- `max_results` caps response size.
- Prefer `OPENALEX_MAILTO` or `UNPAYWALL_EMAIL` for the polite pool.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `openalex`, `arxiv`, `pubmed`, `llm`, `python`
