# OpenAlex Institutions Source Guide

![OpenAlex institutions connector demo](../assets/openalex_institutions_source.gif)

Use this guide when wiring OpenAlex institutions into **scholar-rag-agent**. The agent
can route downstream synthesis through GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 when enabled, but the institutions connector itself is deterministic JSON; no
LLM is required to discover research organizations.

## Why OpenAlex institutions

OpenAlex indexes universities, hospitals, companies, and other research organizations
with display names, geographic metadata, ROR and Wikidata links, and bibliometric
summaries. Alongside OpenAlex authors, works, and concepts, institution search helps
RAG pipelines anchor literature discovery to the right affiliations and funders.

Free-text institution discovery:

```text
GET https://api.openalex.org/institutions?search=harvard&per-page=5
```

When the query is an OpenAlex institution id (`I####`), the connector resolves the
institution directly via:

```text
GET https://api.openalex.org/institutions/I136199984
```

Pass `OpenAlexInstitutionsConnector(mailto=...)` or set `OPENALEX_MAILTO` (falling
back to `UNPAYWALL_EMAIL`) to join OpenAlex's polite pool. Blank queries and
non-positive `max_results` return no documents and do not issue HTTP requests.
Unavailable API responses are treated as empty results.

## What you get

| Field | Source |
|---|---|
| `title` | `display_name` |
| `text` | Synthesized institution descriptor (type, location, works, h-index) |
| `source` | OpenAlex institution `id` |
| `metadata.source_type` | `"openalex_institutions"` |
| `metadata.institution_id` | Bare institution id (for example `I136199984`) |
| `metadata.type` | Institution `type` |
| `metadata.country_code` | `country_code` |
| `metadata.works_count` | `works_count` |
| `metadata.cited_by_count` | `cited_by_count` |
| `metadata.ror` | Linked ROR id when present |
| `metadata.wikidata` | Linked Wikidata Q-id when present |
| `metadata.city` | `geo.city` |
| `metadata.country` | `geo.country` |
| `metadata.h_index` | `summary_stats.h_index` |

## Example

```python
import asyncio

from ingestion.openalex_institutions import OpenAlexInstitutionsConnector

documents = asyncio.run(
    OpenAlexInstitutionsConnector(mailto="dev@example.org").search(
        "harvard",
        max_results=5,
    )
)
for document in documents:
    print(document.metadata["works_count"], document.title)
```

Institution id lookup:

```python
documents = asyncio.run(OpenAlexInstitutionsConnector().search("I136199984", max_results=1))
```

## Safety notes

- Institution metadata is OpenAlex catalog data — verify affiliations against primary
  sources before treating them as authoritative.
- Untitled institution records are skipped rather than raised.
- Prefer frontier models for downstream synthesis over institution summaries:
  **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
