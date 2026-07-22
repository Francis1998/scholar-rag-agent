# OSF Source Guide

![OSF connector demo](../assets/osf_source.gif)

Use this guide when wiring Open Science Framework (OSF) into
**scholar-rag-agent**. The agent can route downstream synthesis through
GPT-5.5 / Claude Sonnet 4.6 / Gemini 2.5 / Kimi K2 when enabled, but the OSF
connector itself is deterministic JSON — no LLM required to list matching
preprints and registrations.

## Why OSF

OSF hosts multidisciplinary open-science projects, preprints, and registered
study records. Alongside Zenodo, Figshare, and OpenAIRE it surfaces transparent
research workflows, preregistrations, and community preprints that may not be
indexed yet by publisher-centric APIs.

Public keyword search:

```
GET https://api.osf.io/v2/preprints/?filter[search]=open+science&page[size]=5
GET https://api.osf.io/v2/registrations/?filter[search]=open+science&page[size]=5
```

`page[size]` is capped at **100** by the connector. The response is JSON:API
with records under `data`. No OSF token is required for public metadata.

## What you get

| Field | Source |
|---|---|
| `title` | `attributes.title` |
| `text` | Collapsed `attributes.description`, or a contributor/category/year descriptor |
| `source` | `links.html`, else `links.iri` / `links.self`, else `https://osf.io/{id}/` |
| `metadata.doi` | `attributes.doi` / DOI-like links / identifiers |
| `metadata.year` | Leading four digits from published, registered, created, or modified dates |
| `metadata.authors` | Embedded contributor display names when OSF returns them |
| `metadata.resource_type` | `"preprint"` or `"registration"` |
| `metadata.source_type` | `"osf"` |

## Example

```python
import asyncio

from ingestion.osf import OsfConnector

documents = asyncio.run(OsfConnector().search("open science replication", max_results=5))
for document in documents:
    print(document.metadata["resource_type"], document.title)
```

## Safety notes

- Public unauthenticated search only — no OSF account token is required.
- Blank queries and non-positive `max_results` short-circuit with no HTTP call.
- OSF network/API errors are treated as empty endpoint results, so optional
  ingestion flows keep running when the service is unavailable.
- Records without a title are skipped rather than raising.
