# Crossref Works Type-and-License Source Guide

![Crossref works type-and-license discovery flow](../assets/crossref_works_type_license_source.gif)

Use this guide when retrieving Crossref works constrained by both resource type
and license metadata. Discovery is deterministic HTTP against the public
Crossref REST API; optional downstream synthesis can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why combine type and license

Type-only searches mix licensed and unlicensed records, while license-only
searches mix journals, proceedings, datasets, and other outputs. This connector
combines both facets and remains distinct from `crossref_types.py` and
`crossref_works_license.py`.

## Usage

```python
from ingestion.crossref_works_type_license import CrossrefWorksTypeLicenseConnector

# Exact type and license URL
docs = await CrossrefWorksTypeLicenseConnector(mailto="dev@example.org").search(
    "journal-article|https://creativecommons.org/licenses/by/4.0",
    max_results=5,
)

# Free text uses type:journal-article,has-license:true by default
docs = await CrossrefWorksTypeLicenseConnector().search(
    "protein design",
    max_results=5,
)

# Select another default type for free-text searches
docs = await CrossrefWorksTypeLicenseConnector(default_type="dataset").search(
    "climate observations",
    max_results=5,
)
```

## Request behavior

| Query | Crossref parameters |
|---|---|
| `journal-article\|https://creativecommons.org/...` | `filter=type:journal-article,license.url:https://creativecommons.org/...` |
| Free text | `query={text}&filter=type:{default_type},has-license:true` |

## What you get

| Field | Source |
|---|---|
| `title` | Crossref work title |
| `text` | JATS-stripped abstract, else a bibliographic descriptor |
| `source` | `https://doi.org/{DOI}` when present |
| `metadata.source_type` | `crossref_works_type_license` |
| `metadata.crossref_type` | Returned work type, falling back to the query type |
| `metadata.license_url` | Queried or first declared license URL |
| `metadata.licenses` | All declared license URLs |
| `metadata.doi` / `year` / `authors` | Normalized bibliographic metadata |

## Safety notes

- Public Crossref API only; optional `mailto` routes polite-pool traffic.
- Blank input and non-positive limits do not issue HTTP requests.
- Work-type slugs are validated before being included in filter expressions.
- Unavailable or malformed responses return an empty list.
