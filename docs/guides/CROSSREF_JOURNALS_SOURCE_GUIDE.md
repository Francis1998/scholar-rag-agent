# Crossref Journals Source Guide

![Crossref journals discovery flow](../assets/crossref_journals_source.gif)

Use this guide when wiring Crossref **journals** into **scholar-rag-agent**.
Discovery is deterministic HTTP against the public Crossref journals API —
enrichment with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 is optional
after documents are collected.

## Why Crossref journals

Crossref indexes journal and serial metadata (title, ISSN, publisher, subjects,
DOI counts). Venue-aware retrieval complements Crossref works/members and
OpenAlex sources connectors.

## Usage

```python
from ingestion.crossref_journals import CrossrefJournalsConnector

docs = await CrossrefJournalsConnector(mailto="dev@example.org").search(
    "Journal of Machine Learning Research",
    max_results=5,
)
```

ISSN-shaped queries such as `1532-4435` resolve directly via
`GET /journals/{issn}`.

## What you get

| Field | Source |
|---|---|
| `title` | Journal `title` |
| `text` | Journal descriptor (publisher, ISSN, subjects, DOI counts) |
| `metadata.source_type` | `crossref_journals` |
| `metadata.issn` | Primary ISSN |
| `metadata.issns` | Comma-joined ISSN list |
| `metadata.publisher` | Publisher name |
| `metadata.subjects` | Comma-joined subjects |
| `metadata.total_dois` | `counts.total-dois` |

## Safety notes

- Public Crossref API only — no authenticated endpoints.
- Failures return an empty list rather than raising.
- `max_results` caps response size.
- Prefer `CROSSREF_MAILTO` or `OPENALEX_MAILTO` for the polite pool.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `crossref`, `journals`, `issn`, `arxiv`, `pubmed`, `llm`, `python`
