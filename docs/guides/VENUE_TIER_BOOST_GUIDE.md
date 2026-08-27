# Venue Tier Boost Guide

![Venue tier boost demo](../assets/venue_tier_boost.gif)

`VenueTierBooster` re-scores retrieved results by blending the previous
relevance score with a venue prestige tier signal. It is inspired by
LlamaIndex/Haystack metadata boost postprocessors, but is fully local and
deterministic from chunk metadata (not a DOI connector). Re-ranked evidence
can then feed GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or Kimi K2.

## How scoring works

Tier scores:

| Tier | Score |
|------|-------|
| `tier1` | `1.0` |
| `tier2` | `0.7` |
| `tier3` | `0.4` |
| unknown | `0.2` |

Resolution order per chunk:

1. Use `metadata["venue_tier"]` when it is `tier1` / `tier2` / `tier3`.
2. Else look up `metadata["venue"]` or `metadata["journal"]` in the merged
   venue map (optional constructor `venue_tiers` overlays a built-in prestige
   list: Nature, Science, Cell, NEJM, Lancet, JAMA → `tier1`).
3. Else unknown `0.2`.

```text
new_score = (1 - alpha) * old_score + alpha * tier_score
```

- Results are sorted by `new_score` descending. Equal scores keep input order
  (stable sort).
- Input `SearchResult` objects are not mutated; returned rows use
  `retriever="venue_tier_boost"` and append the prior retriever to `path`.

This is distinct from `CitationCountBooster` (citation volume) and from
`FreshnessBooster` (publication-date decay).

## Example

```python
from retrieval.venue_tier_boost import VenueTierBooster

booster = VenueTierBooster(
    alpha=0.3,
    venue_tiers={"My Society Journal": "tier2"},
)
boosted = booster.boost(retrieved_results, top_k=10)
```

## Configuration notes

- `alpha` must be a finite number in `[0.0, 1.0]` (default `0.3`).
- `venue_tiers` values must be `tier1`, `tier2`, or `tier3`.
- `alpha=0.0` preserves prior scores (still re-sorts stably and rewrites
  retriever provenance).
- `alpha=1.0` ranks purely by venue tier score.
- `top_k=None` keeps every result after sorting; empty input returns `[]`.
