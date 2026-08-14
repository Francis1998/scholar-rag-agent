# DataCite DOI Prefix Source Guide

![DataCite DOI prefix discovery flow](../assets/datacite_dois_prefix_source.gif)

Use this guide when DataCite discovery must be scoped to a DOI registration
prefix. Discovery is deterministic HTTP against the public DataCite JSON:API;
optional downstream synthesis can use GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

## Why DOI-prefix search

A DOI prefix identifies a registrant's namespace. Prefix filtering supports
repository- or publisher-focused ingestion without replacing general DataCite
search, report filtering, or related-identifier enrichment.

## Usage

```python
from ingestion.datacite_dois_prefix import DataCiteDoisPrefixConnector

# Prefix-shaped input is sent as the prefix filter
docs = await DataCiteDoisPrefixConnector().search(
    "10.5281",
    max_results=5,
)

# Free text can be constrained by a default prefix
docs = await DataCiteDoisPrefixConnector(default_prefix="10.5281").search(
    "retrieval benchmark",
    max_results=5,
)

# Without a default, free text searches the full DataCite registry
docs = await DataCiteDoisPrefixConnector().search(
    "climate observations",
    max_results=5,
)
```

## Request behavior

| Query | DataCite parameters |
|---|---|
| `10.xxxx` | `prefix=10.xxxx` |
| Free text with a default | `query={text}&prefix={default_prefix}` |
| Free text without a default | `query={text}` |

All requests include a bounded `page[size]` and call
`GET https://api.datacite.org/dois`.

## What you get

| Field | Source |
|---|---|
| `title` | Primary DataCite title |
| `text` | Abstract/description, else a bibliographic descriptor |
| `source` | Landing URL or `https://doi.org/{doi}` |
| `metadata.source_type` | `datacite_dois_prefix` |
| `metadata.doi_prefix` | Prefix extracted from the DOI or selected filter |
| `metadata.doi` / `year` / `authors` | Normalized bibliographic metadata |
| `metadata.publisher` / `resource_type` | DataCite publisher and resource type |

## Safety notes

- Public DataCite API only; no credentials are required.
- Prefixes must match `10.` followed by four to nine digits.
- Blank input and non-positive limits do not issue HTTP requests.
- Unavailable or malformed responses return an empty list.
