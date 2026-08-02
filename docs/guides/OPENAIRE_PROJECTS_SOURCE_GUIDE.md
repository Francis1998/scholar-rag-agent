# OpenAIRE Projects Source Guide

![OpenAIRE projects connector demo](../assets/openaire_projects_source.gif)

Use this guide when wiring OpenAIRE funded projects into **scholar-rag-agent**. The agent
can route downstream synthesis through GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x /
Kimi K2 when enabled, but the projects connector itself is deterministic JSON; no
LLM is required to discover grant records.

## Why OpenAIRE projects

OpenAIRE aggregates funded research projects across European and international funder
registries. This connector is projects-only — distinct from `openaire.py`, which indexes
research products. Project search helps RAG pipelines link literature questions to
grant titles, summaries, funder names, and grant identifiers.

Keyword project discovery:

```text
GET https://api.openaire.eu/search/projects?format=json&keywords=machine+learning&page=1&size=5
```

Title-shaped queries use the supported `name` parameter:

```text
GET https://api.openaire.eu/search/projects?format=json&name=quantum+cryptography&page=1&size=5
```

Grant-shaped queries resolve via `grantID` or `openaireProjectID` when the query carries
a funder-scoped identifier. Blank queries and non-positive `max_results` return no
documents and do not issue HTTP requests. Unavailable API responses are treated as empty
results.

## What you get

| Field | Source |
|---|---|
| `title` | Project `title` |
| `text` | Project `summary` abstract, else a synthesized descriptor |
| `source` | `originalId` or grant `code` |
| `metadata.source_type` | `"openaire_projects"` |
| `metadata.project_id` | `originalId` or `code` |
| `metadata.code` | Grant `code` |
| `metadata.abstract` | Project `summary` |
| `metadata.keywords` | Project `keywords` |
| `metadata.funder` | Primary funder `name` from `fundingtree` |
| `metadata.start_date` | `startdate` |
| `metadata.end_date` | `enddate` |
| `metadata.funded_amount` | `fundedamount` |

## Example

```python
import asyncio

from ingestion.openaire_projects import OpenaireProjectsConnector

documents = asyncio.run(
    OpenaireProjectsConnector().search(
        "machine learning",
        max_results=5,
    )
)
for document in documents:
    print(document.metadata["funder"], document.title)
```

Grant id lookup:

```python
documents = asyncio.run(OpenaireProjectsConnector().search("214458", max_results=1))
```

## Safety notes

- Project summaries are funder-registry metadata — verify funding claims against primary
  grant records before treating them as authoritative.
- Untitled project records are skipped rather than raised.
- Prefer frontier models for downstream synthesis over project summaries:
  **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2**.
