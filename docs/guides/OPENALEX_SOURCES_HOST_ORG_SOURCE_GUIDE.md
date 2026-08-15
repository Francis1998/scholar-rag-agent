# OpenAlex Sources Host Organization Source Guide

![OpenAlex sources host-organization discovery flow](../assets/openalex_sources_host_org_source.gif)

Use this guide when venue retrieval needs sources filtered by their OpenAlex host
organization (publisher). Discovery is deterministic HTTP against the public
OpenAlex API; optional downstream synthesis can use GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

## Why host-organization sources

OpenAlex sources represent journals, repositories, conferences, and other
venues. Filtering by host organization surfaces the venues a publisher maintains
without building hierarchy ancestry paths. This connector is distinct from
`openalex_sources_hierarchy.py`, which normalizes `host > type > ISSN > source`
paths.

## Usage

```python
from ingestion.openalex_sources_host_org import OpenAlexSourcesHostOrgConnector

# Bare or URL-shaped publisher/host organization ids
docs = await OpenAlexSourcesHostOrgConnector(mailto="dev@example.org").search(
    "P4310319965",
    max_results=5,
)

# Free text searches publishers, then fetches their hosted sources
docs = await OpenAlexSourcesHostOrgConnector().search(
    "Springer Nature",
    max_results=5,
)
```

`P####` identifiers call `GET https://api.openalex.org/sources?filter=host_organization:https://openalex.org/P####`.
Free text first calls `GET /publishers?search=...`, then filters sources per host.

## What you get

| Field | Source |
|---|---|
| `title` | Source `display_name` |
| `text` | Host organization context followed by the venue descriptor |
| `source` | OpenAlex source URL |
| `metadata.source_type` | `openalex_sources_host_org` |
| `metadata.openalex_source_id` | Bare `S####` source id |
| `metadata.host_organization` | Bare host organization id (`P####`) |
| `metadata.host_organization_name` | Host display name |
| `metadata.host_org_works_count` | Source `works_count` under the host |
| `metadata.type` / `issn_l` / `works_count` | Inherited OpenAlex source fields |

## Safety notes

- Public OpenAlex API only; optional `mailto` routes polite-pool traffic.
- Blank input and non-positive limits do not issue HTTP requests.
- Response size is bounded by `max_results` and OpenAlex's page-size cap.
- Unavailable or malformed responses return an empty list.
