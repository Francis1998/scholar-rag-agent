# arXiv HTML Abstract Source Guide

![arXiv HTML abstract enrichment flow](../assets/arxiv_html_abstract_source.gif)

Use this guide when wiring **arXiv abs HTML abstract enrichment** into
**scholar-rag-agent**. Discovery fetches `https://arxiv.org/abs/{id}` abstract
text (or enriches free-text arXiv API hits). Downstream synthesis with
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 is optional after documents
are collected.

## Why HTML abs abstracts

The Atom API abstracts are usable, but abs HTML pages expose
`citation_abstract` meta tags and `blockquote.abstract` blocks that are useful
for enrichment and cross-checking. This connector prefers HTML abs text when
available and falls back to Atom summaries for free-text search hits.

## Usage

```python
from ingestion.arxiv_html_abstract import ArxivHtmlAbstractConnector

# Direct abs-page fetch for an arXiv id
docs = await ArxivHtmlAbstractConnector().search("2301.00001", max_results=1)

# Free-text: arXiv API search, then enrich first N with HTML abs when available
docs = await ArxivHtmlAbstractConnector().search("graph neural networks", max_results=3)
```

## What you get

| Field | Source |
|---|---|
| `title` | Abs HTML `citation_title` or Atom entry title |
| `text` | HTML abs abstract (preferred) or Atom summary |
| `metadata.source_type` | `arxiv_html_abstract` |
| `metadata.arxiv_id` | Bare arXiv id |
| `metadata.abstract_source` | `html_abs` or `atom_api` |

## Safety notes

- Public arXiv endpoints only — no authenticated APIs.
- Failures return an empty list rather than raising.
- `max_results` caps free-text enrichment.

## Suggested repo metadata

- **Description:** Agentic scholarly RAG with multi-source ingestion, hybrid retrieval, and GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 reasoning.
- **Topics:** `rag`, `scholarly`, `arxiv`, `llm`, `python`
