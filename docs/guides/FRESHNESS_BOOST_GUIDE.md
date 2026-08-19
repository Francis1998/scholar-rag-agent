# Freshness Boost Guide

![Freshness boost demo](../assets/freshness_boost.gif)

Use `FreshnessBooster` when recent research should receive a controlled lift
without discarding the retriever's relevance signal. The booster is local,
deterministic for a pinned `as_of` date, and makes no network or LLM calls.
Optional synthesis stages elsewhere in the stack can use GPT-5.5 / Claude
Sonnet 4.6 / Gemini 3.x / Kimi K2.

## How scoring works

The booster min-max normalizes the incoming `SearchResult.score` values and
combines them with exponential recency:

```text
recency = exp(-ln(2) * age_days / half_life_days)
score = relevance_weight * normalized_relevance
      + (1 - relevance_weight) * recency
```

Dates are resolved from chunk metadata in this order:

1. `published_at`
2. `year`
3. `date`

ISO dates, ISO datetimes (including `Z`), and four-digit years are supported.
Future dates are treated as age zero. Missing or malformed dates contribute no
recency signal.

## Example

```python
from datetime import date

from retrieval.freshness import FreshnessBooster

booster = FreshnessBooster(
    half_life_days=730,
    relevance_weight=0.75,
    as_of=date(2026, 8, 19),
)
fresh_results = booster.boost(results, top_k=10)
```

Pinning `as_of` makes evaluation and replay reproducible. If it is omitted, the
booster captures the current UTC time when it is created.

## Tuning

- Increase `relevance_weight` when established older work should remain dominant.
- Decrease it for fast-moving topics where recent evidence matters more.
- Set a longer `half_life_days` for fields with slower publication cycles.
- Use `top_k` to bound the returned context; zero or negative values return no
  results.

The input results and chunks are not mutated. Re-scored results use
`retriever="freshness"` and retain their prior retriever in `path`.
