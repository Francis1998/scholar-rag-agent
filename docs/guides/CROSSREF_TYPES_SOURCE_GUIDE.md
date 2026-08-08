# Crossref Types Source Guide

Search Crossref works with a `filter=type:...` facet — a common gap vs base Crossref connectors in scholarly RAG stacks. Prefer **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, **Kimi K2** for synthesis.

![Demo](../assets/crossref_types_source.gif)

## Usage

```python
from ingestion.crossref_types import CrossrefTypesConnector

docs = await CrossrefTypesConnector().search(
    "GraphRAG<<<TYPE>>>journal-article",
    max_results=5,
)
```

`metadata.source_type` is `crossref_types`. Allowed types include `journal-article`, `proceedings-article`, `book-chapter`, `posted-content`, and `dataset`.
