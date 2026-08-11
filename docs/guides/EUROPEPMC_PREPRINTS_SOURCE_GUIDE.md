# Europe PMC Preprints Source Guide

![Europe PMC preprints discovery flow](../assets/europepmc_preprints_source.gif)

Use this guide when wiring Europe PMC's preprint-only search into
**scholar-rag-agent**. Discovery is deterministic HTTP against the public
Europe PMC REST API; optional downstream synthesis can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why Europe PMC preprints

Europe PMC assigns indexed preprints to the `PPR` source, spanning life-science
servers such as bioRxiv and medRxiv. A dedicated connector keeps rapid,
not-yet-peer-reviewed findings distinguishable from the mixed publication
sources returned by the general Europe PMC connector.

## Usage

```python
from ingestion.europepmc_preprints import EuropePmcPreprintsConnector

docs = await EuropePmcPreprintsConnector(email="dev@example.org").search(
    "protein design",
    max_results=5,
)
```

The connector sends `({query}) AND SRC:PPR` with `resultType=core`, JSON output,
and a bounded `pageSize`. It also checks every returned record's `source`
client-side and excludes anything other than `PPR`.

## What you get

| Field | Source |
|---|---|
| `title` | Preprint title |
| `text` | Abstract, else an authors/server/year/DOI descriptor |
| `source` | Stable Europe PMC `/article/PPR/{id}` page |
| `metadata.source_type` | `europepmc_preprints` |
| `metadata.preprint_id` | Europe PMC PPR identifier |
| `metadata.doi` | Preprint DOI |
| `metadata.year` | `pubYear`, else `firstPublicationDate` year |
| `metadata.authors` | `authorString` |
| `metadata.preprint_server` | `journalTitle` |
| `metadata.cited_by_count` | Europe PMC citation count |
| `metadata.is_open_access` | Normalized open-access flag |

## Safety notes

- Preprints have generally not completed peer review; treat claims as
  provisional and preserve the preprint label in downstream answers.
- Public Europe PMC API only; an optional email identifies polite traffic.
- Blank input and non-positive limits do not issue HTTP requests.
- Unavailable or malformed responses return an empty list.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `europepmc`, `preprints`, `ppr`, `biorxiv`, `medrxiv`, `llm`, `python`
