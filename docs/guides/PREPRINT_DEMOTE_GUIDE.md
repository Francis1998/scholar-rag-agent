# Preprint Demote Guide

![Preprint demote demo](../assets/preprint_demote.gif)

`PreprintDemoter` soft-demotes retrieved results whose metadata indicates a
preprint venue or publication type. It is inspired by LlamaIndex/Haystack
metadata boost postprocessors, but is fully local and deterministic from chunk
metadata (not a DOI connector). Re-ranked evidence can then feed GPT-5.5,
Claude Sonnet 4.6, Gemini 3.x, or Kimi K2.

## How scoring works

Preprint detection reads `publication_type`, `type`, or `venue` and treats a
hit as preprint when the casefolded value contains any of:

`arxiv`, `biorxiv`, `medrxiv`, `preprint`, `ssrn`.

```text
demote_score = 0.2 if preprint else 1.0
new_score = (1 - alpha) * old_score + alpha * demote_score
```

- Results are sorted by `new_score` descending. Equal scores keep input order
  (stable sort).
- Input `SearchResult` objects are not mutated; returned rows use
  `retriever="preprint_demote"` and append the prior retriever to `path`.

This is distinct from `VenueTierBooster` (prestige tiers) and from
`OpenAccessPreferencer` (OA preference).

## Example

```python
from retrieval.preprint_demote import PreprintDemoter

demoter = PreprintDemoter(alpha=0.25)
demoted = demoter.demote(retrieved_results, top_k=10)
```

## Configuration notes

- `alpha` must be a finite number in `[0.0, 1.0]` (default `0.25`).
- `alpha=0.0` preserves prior scores (still re-sorts stably and rewrites
  retriever provenance).
- `alpha=1.0` ranks purely by demote score (`0.2` vs `1.0`).
- `top_k=None` keeps every result after sorting; empty input returns `[]`.
