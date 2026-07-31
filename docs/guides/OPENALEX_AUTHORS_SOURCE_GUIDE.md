# OpenAlex Authors Source Guide

![OpenAlex authors connector demo](../assets/openalex_authors_source.gif)

Use this guide when wiring OpenAlex authors into **scholar-rag-agent**. The agent
can route downstream synthesis through GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 when enabled, but the authors connector itself is deterministic JSON; no
LLM is required to discover researcher profiles.

## Why OpenAlex authors

OpenAlex indexes researcher profiles with display names, ORCID ids, affiliation
history, and bibliometric summaries. Alongside OpenAlex works, ORCID, and
Semantic Scholar, author search helps RAG pipelines map a researcher name to
their publication footprint before drilling into individual papers.

Free-text author discovery:

```text
GET https://api.openalex.org/authors?search=Geoffrey+Hinton&per-page=5
```

When the query is an OpenAlex author id (`A####`), the connector resolves the
author directly via:

```text
GET https://api.openalex.org/authors/A2208157607
```

Pass `OpenAlexAuthorsConnector(mailto=...)` or set `OPENALEX_MAILTO` (falling
back to `UNPAYWALL_EMAIL`) to join OpenAlex's polite pool. Blank queries and
non-positive `max_results` return no documents and do not issue HTTP requests.
Unavailable API responses are treated as empty results.

## What you get

| Field | Source |
|---|---|
| `title` | `display_name` |
| `text` | Synthesized author descriptor (affiliation, works, citations, h-index) |
| `source` | OpenAlex author `id` |
| `metadata.source_type` | `"openalex_authors"` |
| `metadata.author_id` | Bare author id (for example `A2208157607`) |
| `metadata.orcid` | Bare ORCID id |
| `metadata.works_count` | `works_count` |
| `metadata.cited_by_count` | `cited_by_count` |
| `metadata.institution` | First `last_known_institutions[].display_name` |
| `metadata.h_index` | `summary_stats.h_index` |
| `metadata.i10_index` | `summary_stats.i10_index` |

## Example

```python
import asyncio

from ingestion.openalex_authors import OpenAlexAuthorsConnector

documents = asyncio.run(
    OpenAlexAuthorsConnector(mailto="dev@example.org").search(
        "Geoffrey Hinton",
        max_results=5,
    )
)
for document in documents:
    print(document.metadata["works_count"], document.title)
```

Author id lookup:

```python
documents = asyncio.run(OpenAlexAuthorsConnector().search("A2208157607", max_results=1))
```

## Safety notes

- Author profiles are OpenAlex catalog metadata — verify against primary sources
  before treating bibliometrics as authoritative.
- Untitled author records are skipped rather than raised.
- Prefer frontier models for downstream synthesis over author summaries:
  **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
