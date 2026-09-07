# BibTeX Export Guide

![BibTeX export demo](../assets/bibtex-export.gif)

`BibTeXExporter` converts `Document`, `Chunk`, and `SearchResult` records into
BibTeX bibliography entries using title and common scholarly metadata fields
(`authors`, `year`, `journal`/`venue`, `doi`). Inspired by Zotero / PaperQA
citation export. Pure local transform with no network calls — distinct from
retrieval gates and live DOI connectors. Local exporter for GPT-5.5 / Claude
Sonnet 4.6 / Gemini 3.x / Kimi K2 literature workflows.

## Usage

```python
from retrieval.bibtex_export import BibTeXExporter

exporter = BibTeXExporter()
bib = exporter.export_results(ranked_hits)
# or: exporter.export_documents(docs) / exporter.export_chunks(chunks)
```

Collections deduplicate by `document_id` (highest score wins for search
results). Special characters in titles are escaped for BibTeX safety.
