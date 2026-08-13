# OpenAlex Concepts Ancestors Source Guide

![OpenAlex concepts ancestors discovery flow](../assets/openalex_concepts_ancestors_source.gif)

Use this guide when wiring OpenAlex concept ancestor hierarchy into
**scholar-rag-agent**. Discovery is deterministic HTTP against the public
OpenAlex API; optional downstream synthesis can use GPT-5.5 /
Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Why OpenAlex concepts ancestors

Legacy OpenAlex concepts nest under ancestor concepts by `level`. Leaf-only
concept connectors omit the taxonomy path. This connector fills that gap by
normalizing `ancestors` into `ancestor_path`, distinct from
`openalex_concepts.py` (descriptions / sample works).

## Usage

```python
from ingestion.openalex_concepts_ancestors import OpenAlexConceptsAncestorsConnector

# Direct OpenAlex concept id
docs = await OpenAlexConceptsAncestorsConnector(mailto="dev@example.org").search(
    "C119857082",
    max_results=5,
)

# Free-text concept search
docs = await OpenAlexConceptsAncestorsConnector().search(
    "machine learning",
    max_results=5,
)
```

Concept-id forms accepted: bare `C####`, and `https://openalex.org/C####`.

## What you get

| Field | Source |
|---|---|
| `title` | Concept display name |
| `text` | Ancestor path plus description / counts |
| `source` | OpenAlex concept id URL |
| `metadata.source_type` | `openalex_concepts_ancestors` |
| `metadata.concept_id` | Bare OpenAlex concept id |
| `metadata.level` | Concept level |
| `metadata.ancestors` | Ancestor display names (level-ordered) |
| `metadata.ancestor_path` | `ancestor > ... > leaf` path |
| `metadata.works_count` | Works count |
| `metadata.cited_by_count` | Cited-by count |

## Safety notes

- Public OpenAlex API only; optional `mailto` routes polite-pool traffic.
- Blank input and non-positive limits do not issue HTTP requests.
- Unavailable or malformed responses return an empty list.
- Concepts are a legacy OpenAlex taxonomy; prefer topics for new catalogs.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `openalex`, `concepts`, `taxonomy`, `ancestors`, `llm`, `python`
