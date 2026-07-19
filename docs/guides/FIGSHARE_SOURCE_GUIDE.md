# Figshare Source Guide

![Scholar RAG Agent demo](../assets/demo.gif)

Use this guide when wiring Figshare into **scholar-rag-agent**. The agent can
route enrichment through GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
when enabled, but the Figshare connector itself is deterministic JSON — no LLM
required to list matching research outputs.

## Why Figshare

Figshare is a general-purpose open-research repository that mints DOIs for
figures, datasets, media, papers, posters, presentations, theses, and code.
Alongside Zenodo and OpenAIRE it covers long-tail and institutional deposits
that may never appear in PubMed or Crossref.

Public keyword search:

```
POST https://api.figshare.com/v2/articles/search
{"search_for": "retrieval augmented generation", "page_size": 5}
```

`page_size` is capped at **100**. The response is a JSON array of article
objects.

## What you get

| Field | Source |
|---|---|
| `title` | `title` |
| `text` | HTML-stripped `description`, or a `(year)` descriptor when absent |
| `source` | `url_public_html`, else `https://doi.org/{doi}`, else title |
| `metadata.doi` | `doi` |
| `metadata.year` | First four digits of `published_date` when it matches `^\d{4}` |
| `metadata.source_type` | `"figshare"` |

## Example

```python
import asyncio

from ingestion.figshare import FigshareConnector

documents = asyncio.run(FigshareConnector().search("climate dataset", max_results=5))
for document in documents:
    print(document.metadata["doi"], document.title)
```

## Safety notes

- Public unauthenticated search only — no Figshare account tokens required.
- Blank queries and non-positive `max_results` short-circuit with no HTTP call.
- Articles without a title are skipped rather than raising.
