# Semantic Scholar Bulk Source Guide

Batch-resolve many Semantic Scholar paper ids in one request — a gap vs single-id fetchers in popular scholarly RAG stacks (paper-qa, llama-index). Downstream synthesis works best with **GPT-5.5**, **Claude Sonnet 4.6**, **Gemini 3.x**, and **Kimi K2**.

![Demo](../assets/semantic_scholar_bulk_source.gif)

## Usage

```python
from ingestion.semantic_scholar_bulk import SemanticScholarBulkConnector

docs = await SemanticScholarBulkConnector().search("abc123, DOI:10.1000/xyz", max_results=10)
```

`metadata.source_type` is `semantic_scholar_bulk`.
