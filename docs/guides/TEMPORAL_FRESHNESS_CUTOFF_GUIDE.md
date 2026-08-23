# Temporal Freshness Cutoff Guide

![Temporal freshness cutoff demo](../assets/temporal_freshness_cutoff.gif)

`TemporalFreshnessCutoff` drops retrieved chunks that are older than a
configured maximum age before they ever reach synthesis. It is inspired by
LlamaIndex's `TemporalRetriever` recency filter and Haystack-style metadata
freshness filters, but requires no network call or LLM: age is computed
purely from chunk metadata. Surviving evidence can then be handed to GPT-5.5,
Claude Sonnet 4.6, Gemini 3.x, or Kimi K2 for grounded synthesis.

## How it differs from `FreshnessBooster`

- `FreshnessBooster` **re-scores and re-ranks** every result, blending
  relevance with a continuous exponential recency signal. Nothing is dropped.
- `TemporalFreshnessCutoff` is a **hard boolean gate**: a result either
  passes through untouched (same score, same object) or is removed entirely.
  It runs upstream of ranking-sensitive stages that should never see stale
  evidence at all, regardless of how relevant it scored.

The two are complementary and can be composed, e.g. cutoff first to drop
anything past a hard staleness limit, then boost to prefer newer evidence
among what remains.

## How filtering works

1. For each result, read a publication date from chunk metadata, in priority
   order: `published_at`, `year`, then `date`.
2. If no field parses to a date, the result is **kept** when
   `keep_undated=True` (the default) or **dropped** when `keep_undated=False`.
3. If a date parses, compute its age in days relative to `as_of` (defaults to
   the current UTC time). Ages are clamped to zero, so future-dated chunks
   are never penalized.
4. Keep the result only if `age_days <= max_age_days`. The boundary is
   inclusive.

```text
kept = (no date and keep_undated) or (date parses and age_days <= max_age_days)
```

## Example

```python
from retrieval.temporal_freshness_cutoff import TemporalFreshnessCutoff

cutoff = TemporalFreshnessCutoff(max_age_days=730, keep_undated=True)
fresh_results = cutoff.filter(retrieved_results)
```

Pin `as_of` for reproducible tests and offline evaluation:

```python
from datetime import UTC, datetime

cutoff = TemporalFreshnessCutoff(
    max_age_days=365,
    as_of=datetime(2026, 1, 1, tzinfo=UTC),
)
```

## Normalization and safety

- Accepted date formats: bare four-digit years (`"2024"`), ISO dates
  (`"2024-01-01"`), and ISO datetimes with or without a trailing `Z`
  (`"2024-01-01T00:00:00Z"`).
- Malformed date strings (e.g. `"unknown"`) are treated as undated and fall
  back to the next metadata field, then to `keep_undated`.
- Input order and scores of surviving results are preserved exactly; no
  `SearchResult` is copied or mutated.
- `max_age_days` must be a finite, positive number; invalid values raise
  `ValueError` at construction time.
- Empty input returns an empty list.
