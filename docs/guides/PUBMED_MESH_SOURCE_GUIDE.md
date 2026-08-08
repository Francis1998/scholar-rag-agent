# PubMed MeSH Source Guide

![PubMed MeSH discovery flow](../assets/pubmed_mesh_source.gif)

Use this guide when wiring NCBI **MeSH** descriptors into **scholar-rag-agent**.
Discovery is deterministic HTTP against E-utilities (`db=mesh`) — enrichment
with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 is optional after
documents are collected.

## Why PubMed MeSH

Medical Subject Headings are the controlled vocabulary behind PubMed/MEDLINE
indexing. Descriptor documents (UI, preferred name, tree numbers, scope notes)
support terminology-aware biomedical retrieval alongside article connectors.

## Usage

```python
from ingestion.pubmed_mesh import PubmedMeshConnector

docs = await PubmedMeshConnector().search("diabetes mellitus", max_results=5)
```

Optional `NCBI_API_KEY` (or constructor `api_key`) raises E-utilities rate limits.

## What you get

| Field | Source |
|---|---|
| `title` | Preferred MeSH term (`ds_meshterms[0]`) |
| `text` | Descriptor text (UI, tree numbers, scope note, entry terms) |
| `metadata.source_type` | `pubmed_mesh` |
| `metadata.mesh_ui` | MeSH Unique Identifier (e.g. `D003920`) |
| `metadata.name` | Preferred descriptor name |
| `metadata.tree_numbers` | Comma-joined MeSH tree numbers |

## Safety notes

- Public NCBI E-utilities only.
- Blank queries / API failures return an empty list rather than raising.
- `max_results` caps `esearch` / returned descriptors.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `pubmed`, `mesh`, `biomedical`, `llm`, `python`
