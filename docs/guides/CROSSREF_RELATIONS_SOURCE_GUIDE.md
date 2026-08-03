# Crossref Relations Source Guide

![Crossref relations connector demo](../assets/crossref_relations_source.gif)

Use this guide when wiring Crossref works relation enrichment into
**scholar-rag-agent**. The agent can route downstream synthesis through GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 when enabled, but this connector itself is
deterministic JSON; no LLM is required to discover Crossref relation types.

## Why Crossref relations

Crossref works can carry a `relation` object with typed links such as
`is-referenced-by`, `has-review`, `is-preprint-of`, and similar assertions. This
connector queries the same public works API as `crossref.py` but enriches each
document with those relation type keys for searchable provenance — useful when
tracing reviews, preprints, and cross-work relationships alongside Crossref
members, Event Data, and DataCite related identifiers.

Free-text search (optional polite-pool `mailto`):

```text
GET https://api.crossref.org/works?query=retrieval+augmented&rows=5
```

DOI-shaped queries resolve a single work:

```text
GET https://api.crossref.org/works/10.5555/example
```

Accepted DOI forms include bare DOIs (`10.5555/example`), `doi:` prefixes, and
`https://doi.org/...` URLs. Pass `CrossrefRelationsConnector(mailto=...)` or set
`CROSSREF_MAILTO` (falling back to `OPENALEX_MAILTO`) to join Crossref's polite
pool. Blank queries and non-positive `max_results` return no documents and do not
issue HTTP requests. Unavailable API responses are treated as empty results.

## What you get

| Field | Source |
|---|---|
| `title` | First `title[]` entry |
| `text` | JATS-stripped abstract (or title/year descriptor) plus relation summary |
| `source` | `https://doi.org/{doi}` when a DOI is present |
| `metadata.source_type` | `"crossref_relations"` |
| `metadata.doi` | Work DOI |
| `metadata.year` | Resolved publication year |
| `metadata.relation_types` | Comma-joined keys from the work `relation` object |
| `metadata.relation_count` | Count of related objects across relation type lists |

## Example

```python
import asyncio

from ingestion.crossref_relations import CrossrefRelationsConnector

documents = asyncio.run(
    CrossrefRelationsConnector(mailto="dev@example.org").search(
        "retrieval augmented generation",
        max_results=5,
    )
)
for document in documents:
    print(document.metadata["relation_types"], document.title)

work = asyncio.run(CrossrefRelationsConnector().search("10.5555/example", max_results=1))
```

## Safety notes

- Relation assertions are registry metadata — verify links against landing pages
  before treating them as authoritative relationships.
- Works without a usable title are skipped rather than raised.
- No API key is required for public works search.
- Prefer frontier models for downstream synthesis over raw relation summaries:
  **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
