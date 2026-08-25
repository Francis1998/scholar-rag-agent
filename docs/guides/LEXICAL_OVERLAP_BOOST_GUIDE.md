# Lexical Overlap Boost Guide

![Lexical overlap boost demo](../assets/lexical_overlap_boost.gif)

`LexicalOverlapBooster` re-scores retrieved results by blending the previous
relevance score with Jaccard overlap between the query and each chunk. It is
inspired by hybrid BM25 boost / Haystack keyword-boost stages, but is fully
local and deterministic via `retrieval.sparse.meaningful_terms`. Re-ranked
evidence can then feed GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or Kimi K2.

## How scoring works

For each result:

```text
overlap   = |query_terms ∩ chunk_terms| / |query_terms ∪ chunk_terms|
new_score = (1 - alpha) * old_score + alpha * overlap
```

- Terms come from distinct non-stopword tokens in the query and in the chunk
  title + text.
- When both term sets are empty, `overlap` is `0.0`.
- Results are sorted by `new_score` descending. Equal scores keep input order
  (stable sort).
- Input `SearchResult` objects are not mutated; returned rows use
  `retriever="lexical_overlap_boost"` and append the prior retriever to `path`.

This is distinct from `FreshnessBooster` (publication-date decay) and from
`CorrectiveRagGate` (keep/filter/retry without re-scoring).

## Example

```python
from retrieval.lexical_overlap_boost import LexicalOverlapBooster

booster = LexicalOverlapBooster(alpha=0.3)
boosted = booster.boost(query, retrieved_results, top_k=10)
```

## Configuration notes

- `alpha` must be a finite number in `[0.0, 1.0]` (default `0.3`).
- `alpha=0.0` preserves prior scores (still re-sorts stably and rewrites
  retriever provenance).
- `alpha=1.0` ranks purely by Jaccard overlap.
- `top_k=None` keeps every result after sorting; empty input returns `[]`.
