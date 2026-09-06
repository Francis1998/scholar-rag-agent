# Citation Graph Expansion Guide

![Citation graph expansion demo](../assets/citation-graph.gif)

`CitationGraphIndex` and `CitationGraphExpander` build a local directed
paper-citation graph from chunk metadata and expand seed retrieval hits along
citing / cited edges. Inspired by Semantic Scholar citation graphs and
PaperQA-style related-paper gathering. Distinct from entity co-mention GraphRAG
and from citation-count score boosts. Local expander for GPT-5.5 / Claude Sonnet
4.6 / Gemini 3.x / Kimi K2 pipelines (not a live DOI connector).

## Metadata fields

- Paper id: `doi` / `paper_doi` / `work_doi` (else `document_id`)
- Outgoing cites: `references` / `cites` / `reference_dois` / `cites_dois`
- Incoming cites: `cited_by` / `cited_by_dois` / `citing_dois`

Values may be comma, semicolon, pipe, or whitespace separated. DOI prefixes
(`doi:`, `https://doi.org/`) are normalized.

## Usage

```python
from retrieval.citation_graph import CitationGraphExpander, CitationGraphIndex

index = CitationGraphIndex()
index.index_chunks(corpus_chunks)
expander = CitationGraphExpander(index, hop_decay=0.5, direction="both")
expanded = expander.expand(seed_results, max_hops=1)
```

Each hop multiplies the previous paper score by `hop_decay`. Set
`include_seeds=False` to return only neighbours. See unit tests for edge cases.
