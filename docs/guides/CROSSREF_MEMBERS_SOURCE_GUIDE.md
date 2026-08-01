# Crossref Members Source Guide

![Crossref members connector demo](../assets/crossref_members_source.gif)

Use this guide when wiring Crossref members into **scholar-rag-agent**. The agent
can route downstream synthesis through GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 when enabled, but the members connector itself is deterministic JSON —
no LLM is required to list matching publishers and registrants.

## Why Crossref members

Crossref members are scholarly publishers and content registrants indexed in the
Crossref REST API. Alongside Crossref works, funders, DataCite, OpenAlex, and
Semantic Scholar they supply stable member ids, primary names, locations, DOI
prefix lists, and registration counts — useful context for publisher-aware RAG.

Free-text search (optional polite-pool `mailto`):

```text
GET https://api.crossref.org/members?query=elsevier&rows=5
```

Member-id-shaped queries resolve a single registry record:

```text
GET https://api.crossref.org/members/78
```

Accepted id forms include bare numeric ids (`78`). Pass
`CrossrefMembersConnector(mailto=...)` or set `CROSSREF_MAILTO` (falling back to
`OPENALEX_MAILTO`) to join Crossref's polite pool. Blank queries and non-positive
`max_results` return no documents and do not issue HTTP requests. Unavailable API
responses are treated as empty results.

## What you get

| Field | Source |
|---|---|
| `title` | `primary-name` |
| `text` | Descriptor with location, alternate names, prefixes, and DOI counts |
| `source` | `https://api.crossref.org/members/{id}` |
| `metadata.source_type` | `"crossref_members"` |
| `metadata.member_id` | Crossref member id |
| `metadata.location` | Country / region |
| `metadata.prefixes` | Comma-joined DOI prefixes |
| `metadata.alt_names` | Comma-joined alternate names |
| `metadata.total_dois` | `counts.total-dois` when returned |

## Example

```python
import asyncio

from ingestion.crossref_members import CrossrefMembersConnector

documents = asyncio.run(
    CrossrefMembersConnector(mailto="dev@example.org").search(
        "elsevier",
        max_results=5,
    )
)
for document in documents:
    print(document.metadata["member_id"], document.title, document.metadata["location"])
```

## Safety notes

- Blank queries and non-positive `max_results` short-circuit with no HTTP call.
- Members without a usable primary name are skipped rather than raised.
- No API key is required for public member search.
- Prefer frontier models for downstream publisher-aware synthesis over raw member
  rows: **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
