# OpenAlex Sources Hierarchy Source Guide

![OpenAlex sources hierarchy discovery flow](../assets/openalex_sources_hierarchy_source.gif)

Use this guide when venue retrieval needs publisher/host, source type, and ISSN
ancestry in addition to a flat source profile. Discovery is deterministic HTTP
against the public OpenAlex API; optional downstream synthesis can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why source hierarchy

OpenAlex sources represent journals, repositories, conferences, and other
venues. A normalized `host > type > ISSN > source` path lets retrieval preserve
the organizational and identifier context of each venue. This connector is
distinct from `openalex_sources.py` and the topic taxonomy in
`openalex_topics_hierarchy.py`.

## Usage

```python
from ingestion.openalex_sources_hierarchy import OpenAlexSourcesHierarchyConnector

# Free-text source search
docs = await OpenAlexSourcesHierarchyConnector(mailto="dev@example.org").search(
    "Scientific Reports",
    max_results=5,
)

# Bare or URL-shaped OpenAlex source ids resolve directly
docs = await OpenAlexSourcesHierarchyConnector().search(
    "S196734849",
    max_results=1,
)
```

Free text calls `GET https://api.openalex.org/sources?search=...`; `S####`
identifiers call `GET https://api.openalex.org/sources/{id}`.

## What you get

| Field | Source |
|---|---|
| `title` | Source `display_name` |
| `text` | Normalized hierarchy followed by the venue descriptor |
| `source` | OpenAlex source URL |
| `metadata.source_type` | `openalex_sources_hierarchy` |
| `metadata.openalex_source_id` | Bare `S####` source id |
| `metadata.host_organization` | Bare host organization id |
| `metadata.host_organization_name` | Host display name |
| `metadata.type` | OpenAlex source type |
| `metadata.issn_l` / `issn` | Primary ISSN-L and de-duplicated ISSN list |
| `metadata.ancestry_path` | `host > type > ISSN > source` |
| `metadata.hierarchy_path` | Alias of the normalized ancestry path |

## Safety notes

- Public OpenAlex API only; optional `mailto` routes polite-pool traffic.
- Blank input and non-positive limits do not issue HTTP requests.
- Response size is bounded by `max_results` and OpenAlex's page-size cap.
- Unavailable or malformed responses return an empty list.
