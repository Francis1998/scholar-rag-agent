# Figure Caption Indexer Guide

![Figure caption indexer demo](../assets/figure-caption.gif)

`FigureCaptionIndexer` extracts PDF-like **figure** and **table** captions from
plain text blocks using scholarly caption patterns (`Figure 1: ...`,
`Fig. 2. ...`, `Table 3: ...`).

Fills a PaperQA / Unstructured figure-extraction gap for RAG with a local,
deterministic caption indexer (no LLM or layout model). Indexed captions can
feed GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 grounding. Distinct
from `PdfOcrHook` and from live DOI connectors.

## Usage

```python
from ingestion.figure_caption_indexer import FigureCaptionIndexer

indexer = FigureCaptionIndexer()
captions = indexer.extract(
    "Figure 1: Overview of the retrieval pipeline.\nTable 2. Ablation results on the benchmark.\n"
)
for caption in captions:
    print(caption.kind, caption.number, caption.caption, caption.char_offset)
```

Each result is a `FigureCaption(kind, number, caption, char_offset)` ordered by
appearance. Empty or caption-free text returns an empty list.
