# Title Match Boost Guide

![Title match boost demo](../assets/title_match_boost.gif)

`TitleMatchBooster` re-scores retrieved results by blending the previous
relevance score with Jaccard overlap between the query and each chunk **title
only**. It is inspired by LlamaIndex title-metadata boost and Haystack
keyword-in-title stages, but is fully local and deterministic via
`retrieval.sparse.meaningful_terms`. Unlike `LexicalOverlapBooster`, which
measures overlap over title + text, body terms never affect the boost signal.
Re-ranked evidence can then feed GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or
Kimi K2.

## How scoring works

For each result:

```text
overlap   = |query_terms ∩ title_terms| / |query_terms ∪ title_terms|
new_score = (1 - alpha) * old_score + alpha * overlap
```

- Terms come from distinct non-stopword tokens in the query and in
  `chunk.title` only.
- When both term sets are empty, `overlap` is `0.0`.
- Results are sorted by `new_score` descending. Equal scores keep input order
  (stable sort).
- Input `SearchResult` objects are not mutated; returned rows use
  `retriever="title_match_boost"` and append the prior retriever to `path`.

This is distinct from `LexicalOverlapBooster` (title+text Jaccard) and from
`FreshnessBooster` (publication-date decay).

## Example

```python
from retrieval.title_match_boost import TitleMatchBooster

booster = TitleMatchBooster(alpha=0.3)
boosted = booster.boost(query, retrieved_results, top_k=10)
```

## Configuration notes

- `alpha` must be a finite number in `[0.0, 1.0]` (default `0.3`).
- `alpha=0.0` preserves prior scores (still re-sorts stably and rewrites
  retriever provenance).
- `alpha=1.0` ranks purely by title Jaccard overlap.
- `top_k=None` keeps every result after sorting; empty input returns `[]`.
