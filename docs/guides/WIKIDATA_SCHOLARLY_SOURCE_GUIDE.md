# Wikidata Scholarly Source Guide

![Wikidata scholarly connector demo](../assets/wikidata_scholarly_source.gif)

Use this guide when wiring Wikidata scholarly entities into **scholar-rag-agent**.
The agent can route downstream synthesis through GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 when enabled, but the Wikidata connector itself is
deterministic JSON; no LLM is required to discover scholarly entities.

## Why Wikidata scholarly

Wikidata indexes scholarly articles, journals, books, and related research
entities with structured claims (DOI, publication date, instance-of types). The
connector searches via `wbsearchentities`, enriches hits with `wbgetentities`,
and falls back to a lightweight SPARQL query on the scholarly graph endpoint
when entity search returns no usable matches.

Entity search:

```text
GET https://www.wikidata.org/w/api.php?action=wbsearchentities&search=attention+is+all+you+need&language=en&type=item&limit=5&format=json
```

Scholarly SPARQL-lite fallback:

```text
GET https://query-scholarly.wikidata.org/sparql?query=...
```

QID lookup:

```text
GET https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q210272&props=labels|descriptions|claims|sitelinks&format=json
```

## What you get

| Field | Source |
|---|---|
| `title` | English `labels` value |
| `text` | English `descriptions` value, else a synthesized descriptor |
| `source` | `https://www.wikidata.org/wiki/{QID}` |
| `metadata.source_type` | `"wikidata_scholarly"` |
| `metadata.wikidata_id` | Item id (for example `Q210272`) |
| `metadata.description` | English description |
| `metadata.doi` | Claim `P356` (DOI) when present |
| `metadata.year` | Four-digit year from claim `P577` (publication date) |
| `metadata.scholarly` | `"true"` when instance-of or DOI indicates scholarly content |
| `metadata.wikipedia_title` | English Wikipedia sitelink title when present |

## Example

```python
import asyncio

from ingestion.wikidata_scholarly import WikidataScholarlyConnector

documents = asyncio.run(
    WikidataScholarlyConnector().search(
        "attention is all you need",
        max_results=5,
    )
)
for document in documents:
    print(document.metadata["wikidata_id"], document.title)
```

QID lookup:

```python
documents = asyncio.run(WikidataScholarlyConnector().search("Q210272", max_results=1))
```

## Safety notes

- Wikidata entries are community-curated — verify critical claims against
  primary sources.
- Blank queries and non-positive `max_results` short-circuit with no HTTP call.
- No API key is required for public Wikidata Action API or scholarly SPARQL.
- Prefer frontier models for downstream synthesis over Wikidata summaries:
  **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
