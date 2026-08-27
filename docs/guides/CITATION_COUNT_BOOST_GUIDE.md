# Citation Count Boost Guide

![Citation count boost demo](../assets/citation_count_boost.gif)

`CitationCountBooster` re-scores retrieved results by blending the previous
relevance score with a batch-normalized `log1p(citation_count)` signal. It is
inspired by LlamaIndex metadata boost and Haystack score-boost postprocessors
(not a DOI connector), but is fully local and deterministic from chunk
metadata. Re-ranked evidence can then feed GPT-5.5, Claude Sonnet 4.6,
Gemini 3.x, or Kimi K2.

## How scoring works

For each result:

```text
count     = metadata["citation_count"] or metadata["cited_by_count"] or 0
signal    = log1p(count) / max_batch(log1p(count))   # 0 when all counts are 0
new_score = (1 - alpha) * old_score + alpha * signal
```

- Counts are read as non-negative finite numbers from string metadata.
- Missing, blank, or unparsable values contribute `0`.
- `citation_count` is preferred when both keys are present.
- Results are sorted by `new_score` descending. Equal scores keep input order
  (stable sort).
- Input `SearchResult` objects are not mutated; returned rows use
  `retriever="citation_count_boost"` and append the prior retriever to `path`.

This is distinct from `FreshnessBooster` (publication-date decay) and from
`LexicalOverlapBooster` (query-term Jaccard).

## Example

```python
from retrieval.citation_count_boost import CitationCountBooster

booster = CitationCountBooster(alpha=0.3)
boosted = booster.boost(retrieved_results, top_k=10)
```

## Configuration notes

- `alpha` must be a finite number in `[0.0, 1.0]` (default `0.3`).
- `alpha=0.0` preserves prior scores (still re-sorts stably and rewrites
  retriever provenance).
- `alpha=1.0` ranks purely by normalized `log1p` citation counts.
- `top_k=None` keeps every result after sorting; empty input returns `[]`.
