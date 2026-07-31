# bioRxiv / medRxiv Collections Source Guide

![bioRxiv collections connector demo](../assets/biorxiv_collections_source.gif)

Use this guide when wiring bioRxiv / medRxiv subject collections into
**scholar-rag-agent**. The agent can route downstream synthesis through GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 when enabled, but the collections
connector itself is deterministic JSON; no LLM is required to list matching
preprints.

## Why bioRxiv collections

bioRxiv and medRxiv group preprints into subject categories (for example
`cell biology`, `cancer biology`, `cardiovascular medicine`). The public details
API returns a `collection` array of preprint metadata and supports server-side
category filtering on date-range requests. This connector complements the
general `BioRxivConnector` by targeting category-shaped queries and the API's
`collection` response field.

Category-filtered discovery:

```text
GET https://api.biorxiv.org/details/biorxiv/2025-06-01/2025-07-01?category=cell_biology
```

Recent-post window (client-side title/abstract filter):

```text
GET https://api.biorxiv.org/details/biorxiv/100
```

DOI lookup:

```text
GET https://api.biorxiv.org/details/biorxiv/10.1101/339747/na
```

## What you get

| Field | Source |
|---|---|
| `title` | `title` |
| `text` | Collapsed `abstract`, else an author/category/year descriptor |
| `source` | `https://www.{server}.org/content/{doi}` |
| `metadata.source_type` | `"biorxiv_collections"` |
| `metadata.doi` | `doi` |
| `metadata.year` | Leading four digits of `date` |
| `metadata.authors` | `authors` |
| `metadata.category` | `category` |
| `metadata.server` | `server` (`biorxiv` or `medrxiv`) |

## Example

```python
import asyncio

from ingestion.biorxiv_collections import BioRxivCollectionsConnector

documents = asyncio.run(
    BioRxivCollectionsConnector().search(
        "cell biology",
        max_results=5,
        server="biorxiv",
    )
)
for document in documents:
    print(document.metadata["category"], document.title)
```

## Safety notes

- Preprints are unrefereed — verify claims against peer-reviewed literature.
- Blank queries and non-positive `max_results` short-circuit with no HTTP call.
- No API key is required for the public bioRxiv / medRxiv details API.
- Prefer frontier models for downstream synthesis over raw preprint metadata:
  **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
