# Crossref Works-by-License Source Guide

![Crossref works-by-license discovery flow](../assets/crossref_works_license_source.gif)

Use this guide when wiring Crossref licensed-works search into
**scholar-rag-agent**. Discovery is deterministic HTTP against the public
Crossref REST API; optional downstream synthesis can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why Crossref works by license

Crossref license metadata links published works to license URLs (Creative
Commons and publisher licenses). A dedicated works-by-license connector lets
open-access-aware RAG retrieve outputs that declare a license, complementing
the general Crossref works connector.

## Usage

```python
from ingestion.crossref_works_license import CrossrefWorksLicenseConnector

# License-URL shaped queries apply filter=license.url:{url}
docs = await CrossrefWorksLicenseConnector(mailto="dev@example.org").search(
    "https://creativecommons.org/licenses/by/4.0",
    max_results=5,
)

# Free-text queries search works that carry license metadata
docs = await CrossrefWorksLicenseConnector().search(
    "protein design",
    max_results=5,
)
```

## What you get

| Field | Source |
|---|---|
| `title` | Crossref work title |
| `text` | JATS-stripped abstract, else authors/year/licenses/DOI descriptor |
| `source` | `https://doi.org/{DOI}` when present |
| `metadata.source_type` | `crossref_works_license` |
| `metadata.doi` | Work DOI |
| `metadata.year` | `published` / `issued` / print / online year |
| `metadata.authors` | Crossref author names |
| `metadata.license_url` | Query license URL or first declared license |
| `metadata.licenses` | Declared license URLs |

## Safety notes

- Public Crossref API only; optional `mailto` routes polite-pool traffic.
- Blank input and non-positive limits do not issue HTTP requests.
- Unavailable or malformed responses return an empty list.
- License URLs are depositor-supplied metadata, not legal advice.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `crossref`, `license`, `open-access`, `doi`, `llm`, `python`
