# DataCite Related Identifiers Source Guide

![DataCite related identifiers connector demo](../assets/datacite_related_source.gif)

Use this guide when wiring DataCite related-identifier enrichment into
**scholar-rag-agent**. The agent can route downstream synthesis through GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 when enabled, but this connector itself is
deterministic JSON; no LLM is required to discover related DOI links.

## Why DataCite related identifiers

DataCite registers DOIs for research data, software, and other scholarly outputs.
This connector is distinct from `datacite.py`: it queries the same `dois` endpoint
but enriches each record with `relatedIdentifiers` from attributes — version links,
companion datasets, citations to other DOIs, and alternate identifiers. That helps
RAG pipelines trace dataset versions and cross-repository relationships.

Free-text DOI discovery:

```text
GET https://api.datacite.org/dois?query=climate&page[size]=5
```

Blank queries and non-positive `max_results` return no documents and do not issue HTTP
requests. Unavailable API responses are treated as empty results.

## What you get

| Field | Source |
|---|---|
| `title` | Primary `titles[].title` |
| `text` | Abstract description plus related-identifier summary |
| `source` | `attributes.url`, else `https://doi.org/{doi}` |
| `metadata.source_type` | `"datacite_related"` |
| `metadata.doi` | Record `doi` |
| `metadata.year` | `publicationYear` |
| `metadata.authors` | Joined creator names |
| `metadata.publisher` | `publisher` |
| `metadata.resource_type` | `types.resourceTypeGeneral` |
| `metadata.related_identifiers` | Semicolon-joined `relatedIdentifiers` summary |
| `metadata.related_identifier_count` | Count of parsed related identifiers |

## Example

```python
import asyncio

from ingestion.datacite_related import DataciteRelatedConnector

documents = asyncio.run(
    DataciteRelatedConnector().search(
        "climate dataset",
        max_results=5,
    )
)
for document in documents:
    print(document.metadata["related_identifiers"], document.title)
```

## Safety notes

- Related identifiers are registry metadata — verify links against landing pages before
  treating them as authoritative relationships.
- Records without titles are skipped rather than raised.
- Prefer frontier models for downstream synthesis over related-identifier summaries:
  **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
