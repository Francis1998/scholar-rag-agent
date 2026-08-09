# OpenAlex Keywords Source Guide

![OpenAlex keywords discovery flow](../assets/openalex_keywords_source.gif)

Use this guide when wiring OpenAlex **keywords** into **scholar-rag-agent**.
Discovery is deterministic HTTP against the public OpenAlex keywords API —
enrichment with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 is optional
after documents are collected.

## Why OpenAlex keywords

OpenAlex keywords tag scholarly works with human-readable research themes plus
works and citation counts. Keyword-aware retrieval complements topics, concepts,
and paper-first connectors.

## Usage

```python
from ingestion.openalex_keywords import OpenAlexKeywordsConnector

docs = await OpenAlexKeywordsConnector(mailto="dev@example.org").search(
    "machine learning",
    max_results=5,
)
```

OpenAlex keyword URLs such as `https://openalex.org/keywords/machine-learning`
and prefixed ids (`keywords/machine-learning`) resolve directly.

## What you get

| Field | Source |
|---|---|
| `title` | `display_name` |
| `text` | Keyword descriptor (works count, citation count) |
| `metadata.source_type` | `openalex_keywords` |
| `metadata.openalex_keyword_id` | Keyword slug (for example `machine-learning`) |
| `metadata.openalex` | OpenAlex keyword URL/id |
| `metadata.works_count` | `works_count` |
| `metadata.cited_by_count` | `cited_by_count` |
| `metadata.works_api_url` | Filtered works API URL |

## Safety notes

- Public OpenAlex API only — no authenticated endpoints.
- Failures return an empty list rather than raising.
- `max_results` caps response size.
- Prefer `OPENALEX_MAILTO` or `UNPAYWALL_EMAIL` for the polite pool.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `openalex`, `keywords`, `arxiv`, `pubmed`, `llm`, `python`
