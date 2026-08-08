# OpenAlex Funders Source Guide

![OpenAlex funders discovery flow](../assets/openalex_funders_source.gif)

Use this guide when wiring OpenAlex **funders** into **scholar-rag-agent**.
Discovery is deterministic HTTP against the public OpenAlex funders API —
enrichment with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 is optional
after documents are collected.

## Why OpenAlex funders

OpenAlex indexes funding organizations with public identifiers, country
metadata, grant counts, works counts, and citation summaries. Funder-aware
retrieval complements paper-first, institution, venue, and publisher connectors.

## Usage

```python
from ingestion.openalex_funders import OpenAlexFundersConnector

docs = await OpenAlexFundersConnector(mailto="dev@example.org").search(
    "National Science Foundation",
    max_results=5,
)
```

Bare OpenAlex funder ids such as `F4320306076` and OpenAlex funder URLs resolve
directly.

## What you get

| Field | Source |
|---|---|
| `title` | `display_name` |
| `text` | Funder descriptor (description/summary, country, grants, works, citations) |
| `metadata.source_type` | `openalex_funders` |
| `metadata.openalex_funder_id` | Bare `F####` id |
| `metadata.openalex` | OpenAlex funder URL/id from `ids.openalex` or `id` |
| `metadata.ror` | Bare ROR id from `ids.ror` |
| `metadata.wikidata` | Bare Wikidata Q-id from `ids.wikidata` |
| `metadata.country` | Funder country/country code |

## Safety notes

- Public OpenAlex API only — no authenticated endpoints.
- Failures return an empty list rather than raising.
- `max_results` caps response size.
- Prefer `OPENALEX_MAILTO` or `UNPAYWALL_EMAIL` for the polite pool.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `openalex`, `funders`, `arxiv`, `pubmed`, `llm`, `python`
