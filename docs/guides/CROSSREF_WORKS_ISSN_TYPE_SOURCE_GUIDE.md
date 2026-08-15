# Crossref Works ISSN-and-Type Source Guide

![Crossref works ISSN-and-type discovery flow](../assets/crossref_works_issn_type_source.gif)

Use this guide when retrieving Crossref works constrained by both ISSN and
resource type. Discovery is deterministic HTTP against the public Crossref REST
API; optional downstream synthesis can use GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2.

## Why combine ISSN and type

ISSN-only searches mix articles, proceedings, datasets, and other outputs,
while type-only searches span many serials. This connector combines both facets
and remains distinct from `crossref_types.py` and `crossref_journals.py`.

## Usage

```python
from ingestion.crossref_works_issn_type import CrossrefWorksIssnTypeConnector

# Exact ISSN and work type
docs = await CrossrefWorksIssnTypeConnector(mailto="dev@example.org").search(
    "1532-4435|journal-article",
    max_results=5,
)

# Free text uses type:journal-article,has-issn:true by default
docs = await CrossrefWorksIssnTypeConnector().search(
    "protein design",
    max_results=5,
)

# Select another default type for free-text searches
docs = await CrossrefWorksIssnTypeConnector(default_type="dataset").search(
    "climate observations",
    max_results=5,
)
```

## Request behavior

| Query | Crossref parameters |
|---|---|
| `1532-4435\|journal-article` | `filter=issn:1532-4435,type:journal-article` |
| Free text | `query={text}&filter=type:{default_type},has-issn:true` |

## What you get

| Field | Source |
|---|---|
| `title` | Crossref work title |
| `text` | JATS-stripped abstract, else a bibliographic descriptor |
| `source` | `https://doi.org/{DOI}` when present |
| `metadata.source_type` | `crossref_works_issn_type` |
| `metadata.crossref_type` | Returned work type, falling back to the query type |
| `metadata.issn` | Queried ISSN or first declared ISSN on the work |
| `metadata.doi` / `year` / `authors` | Normalized bibliographic metadata |

## Safety notes

- Public Crossref API only; optional `mailto` routes polite-pool traffic.
- Blank input and non-positive limits do not issue HTTP requests.
- ISSN and work-type slugs are validated before being included in filter expressions.
- Unavailable or malformed responses return an empty list.
