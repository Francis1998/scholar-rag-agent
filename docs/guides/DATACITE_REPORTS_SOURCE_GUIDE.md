# DataCite Reports Source Guide

![DataCite reports discovery flow](../assets/datacite_reports_source.gif)

Use this guide when wiring DataCite research-report DOIs into
**scholar-rag-agent**. Discovery is deterministic HTTP against the public
DataCite JSON:API; optional downstream synthesis can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why DataCite reports

DataCite registers grey-literature reports alongside datasets and software.
Filtering DOI search with `resource-type-id=report` keeps technical and research
reports distinct from the broader DataCite registry connector and from Event
Data relationship records.

## Usage

```python
from ingestion.datacite_reports import DataCiteReportsConnector

docs = await DataCiteReportsConnector().search(
    "climate assessment",
    max_results=5,
)
```

The connector calls
`GET https://api.datacite.org/dois?query=...&resource-type-id=report` with a
bounded `page[size]`, then normalizes report DOI attributes.

## What you get

| Field | Source |
|---|---|
| `title` | Primary DataCite title |
| `text` | Abstract/description, else authors/publisher/year/DOI descriptor |
| `source` | Landing `url` or `https://doi.org/{doi}` |
| `metadata.source_type` | `datacite_reports` |
| `metadata.doi` | Report DOI |
| `metadata.year` | `publicationYear` |
| `metadata.authors` | Creator names |
| `metadata.publisher` | Publisher name |
| `metadata.resource_type` | Specific or general resource type |

## Safety notes

- Public DataCite API only; no credentials are required.
- Blank input and non-positive limits do not issue HTTP requests.
- Unavailable or malformed responses return an empty list.
- Report DOIs vary in peer-review status; preserve provenance in answers.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `datacite`, `reports`, `doi`, `grey-literature`, `llm`, `python`
