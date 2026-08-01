# OpenAlex Concepts Source Guide

![OpenAlex concepts connector demo](../assets/openalex_concepts_source.gif)

Use this guide when wiring OpenAlex concepts into **scholar-rag-agent**. The agent
can route downstream synthesis through GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 when enabled, but the concepts connector itself is deterministic JSON; no
LLM is required to discover research themes.

## Why OpenAlex concepts

OpenAlex clusters scholarly works into a legacy concepts taxonomy with
human-readable names, descriptions, Wikidata links, and coverage statistics.
Alongside OpenAlex topics, works, Semantic Scholar, and Crossref, concept search
helps RAG pipelines map a broad question to the right thematic neighborhood
before drilling into papers.

Free-text concept discovery:

```text
GET https://api.openalex.org/concepts?search=machine+learning&per-page=5
```

When the query is an OpenAlex concept id (`C####`), the connector resolves the
concept directly and samples representative works via:

```text
GET https://api.openalex.org/concepts/C119857082
GET https://api.openalex.org/works?filter=concepts.id:119857082&per-page=5
```

Pass `OpenAlexConceptsConnector(mailto=...)` or set `OPENALEX_MAILTO` (falling
back to `UNPAYWALL_EMAIL`) to join OpenAlex's polite pool. Blank queries and
non-positive `max_results` return no documents and do not issue HTTP requests.
Unavailable API responses are treated as empty results.

## What you get

| Field | Source |
|---|---|
| `title` | `display_name` |
| `text` | `description`, else a synthesized concept descriptor |
| `source` | OpenAlex concept `id` |
| `metadata.source_type` | `"openalex_concepts"` |
| `metadata.concept_id` | Bare concept id (for example `C119857082`) |
| `metadata.description` | Concept `description` |
| `metadata.works_count` | `works_count` |
| `metadata.cited_by_count` | `cited_by_count` |
| `metadata.level` | Taxonomy `level` |
| `metadata.wikidata` | Linked Wikidata Q-id when present |
| `metadata.sample_work_titles` | Semicolon-joined titles from filtered works when the query is a concept id |

## Example

```python
import asyncio

from ingestion.openalex_concepts import OpenAlexConceptsConnector

documents = asyncio.run(
    OpenAlexConceptsConnector(mailto="dev@example.org").search(
        "machine learning",
        max_results=5,
    )
)
for document in documents:
    print(document.metadata["works_count"], document.title)
```

Concept id lookup:

```python
documents = asyncio.run(OpenAlexConceptsConnector().search("C119857082", max_results=3))
```

## Safety notes

- Concept descriptions are OpenAlex catalog metadata — verify against primary
  literature before treating them as authoritative domain definitions.
- Untitled concept records are skipped rather than raised.
- Prefer frontier models for downstream synthesis over concept summaries:
  **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
