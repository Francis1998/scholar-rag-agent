# Authority Boost Guide

![Authority Boost](../assets/authority_boost.gif)

Inspired by LlamaIndex/Haystack metadata boost postprocessors; soft-boosts hits
using authority signals in chunk metadata (not a DOI connector).

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 answer stacks.

## Distinct from existing boosters

| Booster | Metadata keys | Missing-signal behavior |
| --- | --- | --- |
| `VenueTierBooster` | `venue_tier`, `venue`, `journal` prestige tiers | unknown → `0.2` |
| `CitationCountBooster` | `citation_count`, `cited_by_count` (batch log1p) | missing → `0.0` |
| `AuthorityBooster` | `source_authority`, `venue_rank`, `is_peer_reviewed`, `impact_factor` | missing → neutral `0.5` |

## Resolution order

1. If `source_authority` is a finite float in `[0.0, 1.0]`, use it directly.
2. Else average any available soft signals:
   - `venue_rank` (1 = best): `max(0, 1 - (rank - 1) * 0.1)`
   - `is_peer_reviewed`: truthy → `1.0`, falsey → `0.2`
   - `impact_factor`: numeric buckets `≥10 → 1.0`, `≥3 → 0.65`, `>0 → 0.35`,
     or labels `high` / `medium` / `low`
3. If none are present → `0.5`.

```text
new_score = (1 - alpha) * old_score + alpha * authority
```

Results are sorted by `new_score` descending (stable). Inputs are not mutated;
returned rows use `retriever="authority_boost"`.

## Usage

```python
from retrieval.authority_boost import AuthorityBooster

processor = AuthorityBooster(alpha=0.3)
boosted = processor.boost(retrieved_results, top_k=10)
```

## Safety

Local chunk metadata only — no network calls and no DOI connector side effects.
