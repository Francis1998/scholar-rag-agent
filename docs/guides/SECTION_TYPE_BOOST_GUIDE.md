# Section Type Boost Guide

![Section type boost demo](../assets/section_type_boost.gif)

`SectionTypeBooster` re-scores retrieved results by blending the previous
relevance score with a preferred section-type signal. It is inspired by
LlamaIndex/Haystack metadata boost postprocessors, but is fully local and
deterministic from chunk metadata (not a DOI connector). Re-ranked evidence
can then feed GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or Kimi K2.

## How scoring works

Default section scores:

| Section | Score |
|---------|-------|
| `results` | `1.0` |
| `methods` | `1.0` |
| `conclusion` | `1.0` |
| `abstract` | `1.0` |
| unknown | `0.2` |

Resolution order per chunk:

1. Look up `metadata["section"]` in the merged section-score map.
2. Else look up `metadata["section_type"]`.
3. Else unknown `0.2`.

An optional constructor `section_scores` map overlays the defaults
(case-insensitive keys; values in `[0.0, 1.0]`).

```text
new_score = (1 - alpha) * old_score + alpha * section_score
```

- Results are sorted by `new_score` descending. Equal scores keep input order
  (stable sort).
- Input `SearchResult` objects are not mutated; returned rows use
  `retriever="section_type_boost"` and append the prior retriever to `path`.

This is distinct from `VenueTierBooster` (venue prestige) and from
`LexicalOverlapBooster` (query-term Jaccard).

## Example

```python
from retrieval.section_type_boost import SectionTypeBooster

booster = SectionTypeBooster(
    alpha=0.3,
    section_scores={"discussion": 0.8},
)
boosted = booster.boost(retrieved_results, top_k=10)
```

## Configuration notes

- `alpha` must be a finite number in `[0.0, 1.0]` (default `0.3`).
- `section_scores` values must be finite numbers in `[0.0, 1.0]`.
- `alpha=0.0` preserves prior scores (still re-sorts stably and rewrites
  retriever provenance).
- `alpha=1.0` ranks purely by section score.
- `top_k=None` keeps every result after sorting; empty input returns `[]`.
