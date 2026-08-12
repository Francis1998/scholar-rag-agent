# Crossref Works-by-Funder Source Guide

![Crossref works-by-funder discovery flow](../assets/crossref_works_funder_source.gif)

Use this guide when wiring Crossref funded-works search into
**scholar-rag-agent**. Discovery is deterministic HTTP against the public
Crossref REST API; optional downstream synthesis can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why Crossref works by funder

Crossref funding metadata links published works to Open Funder Registry ids
(`10.13039/...`). A dedicated works-by-funder connector lets grant-aware RAG
retrieve outputs acknowledged under a funder, complementing the Funder Registry
entity connector (`crossref_funder`).

## Usage

```python
from ingestion.crossref_works_funder import CrossrefWorksFunderConnector

# Funder-id shaped queries apply filter=funder:{id}
docs = await CrossrefWorksFunderConnector(mailto="dev@example.org").search(
    "10.13039/100000001",
    max_results=5,
)

# Free-text queries search works that carry funding acknowledgements
docs = await CrossrefWorksFunderConnector().search(
    "protein design",
    max_results=5,
)
```

Funder-id forms accepted: bare digits (`100000001`), `10.13039/{id}`,
`doi:10.13039/{id}`, and `https://doi.org/10.13039/{id}`.

## What you get

| Field | Source |
|---|---|
| `title` | Crossref work title |
| `text` | JATS-stripped abstract, else authors/year/funders/DOI descriptor |
| `source` | `https://doi.org/{DOI}` when present |
| `metadata.source_type` | `crossref_works_funder` |
| `metadata.doi` | Work DOI |
| `metadata.year` | `published` / `issued` / print / online year |
| `metadata.authors` | Crossref author names |
| `metadata.funder_id` | Query funder id or first acknowledged funder |
| `metadata.funders` | Acknowledged funder names / DOIs |

## Safety notes

- Public Crossref API only; optional `mailto` routes polite-pool traffic.
- Blank input and non-positive limits do not issue HTTP requests.
- Unavailable or malformed responses return an empty list.
- Funding acknowledgements are depositor-supplied metadata, not grant ledgers.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `crossref`, `funder`, `grants`, `doi`, `llm`, `python`
