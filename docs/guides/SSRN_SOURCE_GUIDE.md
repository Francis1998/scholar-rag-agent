# SSRN Preprint Source Guide

![SSRN preprint connector demo](../assets/ssrn_source.gif)

Use this guide when wiring SSRN preprints into **scholar-rag-agent**. The agent
can route downstream synthesis through GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 when enabled, but the SSRN connector itself is deterministic Crossref JSON
— no LLM is required to discover matching abstracts.

## Why SSRN via Crossref

Social Science Research Network (SSRN) preprints are indexed in Crossref under
the `10.2139` DOI prefix. Alongside Crossref works, OSF, and bioRxiv/medRxiv,
this bridge supplies economics, finance, law, and social-science preprint
metadata without scraping SSRN directly.

Free-text bibliographic search (optional polite-pool `mailto`):

```text
GET https://api.crossref.org/works?query=corporate+governance&filter=prefix:10.2139&rows=5
```

SSRN DOI-shaped queries resolve a single work:

```text
GET https://api.crossref.org/works/10.2139/ssrn.3537853
```

Accepted DOI forms include bare ids (`10.2139/ssrn.3537853`), `doi:` prefixes,
and DOI URLs. Pass `SsrnConnector(mailto=...)` or set `CROSSREF_MAILTO` (falling
back to `OPENALEX_MAILTO`) to join Crossref's polite pool. Blank queries and
non-positive `max_results` return no documents and do not issue HTTP requests.
Unavailable API responses are treated as empty results.

## What you get

| Field | Source |
|---|---|
| `title` | Crossref work `title` |
| `text` | JATS-stripped `abstract`, else a synthesized descriptor |
| `source` | `https://doi.org/{doi}` |
| `metadata.source_type` | `"ssrn"` |
| `metadata.doi` | SSRN DOI (`10.2139/...`) |
| `metadata.year` | Publication year from `published` / `issued` |
| `metadata.authors` | Comma-joined author names |
| `metadata.container` | `container-title` when present |
| `metadata.ssrn_url` | SSRN abstract landing URL when present |

## Example

```python
import asyncio

from ingestion.ssrn import SsrnConnector

documents = asyncio.run(
    SsrnConnector(mailto="dev@example.org").search(
        "corporate governance",
        max_results=5,
    )
)
for document in documents:
    print(document.metadata["doi"], document.title)
```

DOI lookup:

```python
documents = asyncio.run(
    SsrnConnector().search("10.2139/ssrn.3537853", max_results=1)
)
```

## Safety notes

- SSRN metadata is Crossref-deposited — verify abstracts against the live SSRN
  page before treating them as authoritative.
- Works outside the `10.2139` prefix are filtered out on single-DOI lookups.
- Prefer frontier models for downstream synthesis over raw preprint metadata:
  **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
