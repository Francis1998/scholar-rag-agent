# ORCID Works Summaries Source Guide

![ORCID works summaries discovery flow](../assets/orcid_works_summaries_source.gif)

Use this guide when wiring ORCID iD → public work summaries into
**scholar-rag-agent**. Discovery is deterministic HTTP against the public
ORCID API; optional downstream synthesis can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why ORCID works summaries

ORCID profiles expose curated work summaries (title, type, journal, put-code,
external ids) under `/{orcid}/works`. This connector is ORCID-iD-first and
summary-oriented — distinct from keyword profile search (`orcid.py`),
year/type deep filters (`orcid_works_filter.py`), and employments
(`orcid_employments.py`).

## Usage

```python
from ingestion.orcid_works_summaries import OrcidWorksSummariesConnector

docs = await OrcidWorksSummariesConnector().search(
    "0000-0002-1825-0097",
    max_results=5,
)

docs = await OrcidWorksSummariesConnector().search(
    "https://orcid.org/0000-0002-1825-0097",
    max_results=5,
)
```

Non-ORCID queries return an empty list without calling the API.

## What you get

| Field | Source |
|---|---|
| `title` | Work summary title |
| `text` | Summary descriptor with type/journal/year/DOI/external ids |
| `source` | Work URL, DOI URL, or ORCID work put-code URL |
| `metadata.source_type` | `orcid_works_summaries` |
| `metadata.orcid` | ORCID iD |
| `metadata.doi` | Preferred DOI external id |
| `metadata.year` | Publication year |
| `metadata.journal` | Journal title |
| `metadata.work_type` | ORCID work type |
| `metadata.put_code` | Work put-code |
| `metadata.external_ids` | Joined `type:value` external identifiers |

## Safety notes

- Public ORCID API only (`Accept: application/json`).
- Blank, non-ORCID, and non-positive limits do not issue HTTP requests.
- Unavailable or malformed responses return an empty list.
- Summaries are author-curated public metadata, not full-text.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `orcid`, `works`, `identifiers`, `llm`, `python`
