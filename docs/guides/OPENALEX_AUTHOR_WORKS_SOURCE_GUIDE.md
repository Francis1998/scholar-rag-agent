# OpenAlex Author Works Source Guide

![OpenAlex author works discovery flow](../assets/openalex_author_works_source.gif)

Use this guide when wiring OpenAlex author → works citation blends into
**scholar-rag-agent**. Discovery is deterministic HTTP against the public
OpenAlex API; optional downstream synthesis can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why OpenAlex author works

Author profiles expose aggregate bibliometrics, but grounded literature review
needs the underlying works with per-work `cited_by_count`. This connector fills
the OpenAlex authors citations blend gap by resolving an author id (or name
search) and fetching:

`GET https://api.openalex.org/works?filter=authorships.author.id:{id}`

sorted by `cited_by_count:desc`.

## Usage

```python
from ingestion.openalex_author_works import OpenAlexAuthorWorksConnector

# Direct OpenAlex author id
docs = await OpenAlexAuthorWorksConnector(mailto="dev@example.org").search(
    "A2208157607",
    max_results=5,
)

# Free-text author name resolves the top author, then fetches works
docs = await OpenAlexAuthorWorksConnector().search(
    "Ada Lovelace",
    max_results=5,
)
```

Author-id forms accepted: bare `A####`, and `https://openalex.org/A####`.

## What you get

| Field | Source |
|---|---|
| `title` | Work title / display name |
| `text` | Reconstructed abstract, else authors/journal/year/citations/DOI descriptor |
| `source` | OpenAlex work id (preferred) or DOI URL |
| `metadata.source_type` | `openalex_author_works` |
| `metadata.author_id` | Resolved OpenAlex author id |
| `metadata.doi` | Bare DOI when present |
| `metadata.year` | Publication year |
| `metadata.authors` | Authorship display names |
| `metadata.journal` | Primary location source name |
| `metadata.openalex_id` | Work OpenAlex URL/id |
| `metadata.cited_by_count` | Per-work citation count |
| `metadata.landing_url` | Primary landing page when known |

## Safety notes

- Public OpenAlex API only; optional `mailto` routes polite-pool traffic.
- Blank input and non-positive limits do not issue HTTP requests.
- Unavailable or malformed responses return an empty list.
- Citation counts are snapshot bibliometrics, not quality judgments.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `openalex`, `authors`, `citations`, `bibliometrics`, `llm`, `python`
