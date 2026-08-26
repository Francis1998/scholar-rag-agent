# Near Duplicate Collapse Guide

![Near duplicate collapse demo](../assets/near_duplicate_collapse.gif)

`NearDuplicateCollapser` drops near-duplicate retrieved chunks so the context
window is not wasted on redundant passages. It is inspired by LlamaIndex
`SimilarityPostprocessor` / dedupe stages, but is fully local and deterministic
via Jaccard similarity over `retrieval.sparse.meaningful_terms` on chunk
**text** (not title). Unlike `MMRDiversifier`, which re-ranks for novelty
without removing rows, this stage hard-collapses redundant hits. Surviving
evidence can then feed GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or Kimi K2.

## How collapse works

1. Sort candidates by score descending (stable for ties).
2. Keep a candidate when its text-term Jaccard similarity to every already-kept
   representative is strictly below `threshold`.
3. Otherwise drop it as a near-duplicate of a higher-scoring hit.
4. Optionally truncate to `top_k` after collapse.

```text
similarity = |terms(a) ∩ terms(b)| / |terms(a) ∪ terms(b)|
```

- Two empty term sets (e.g. stopword-only text) are treated as similarity
  `1.0` and collapse together.
- Input `SearchResult` objects are not mutated; survivors keep identity,
  score, and relative score order.

## Example

```python
from retrieval.near_duplicate_collapse import NearDuplicateCollapser

collapser = NearDuplicateCollapser(threshold=0.9)
deduped = collapser.collapse(retrieved_results, top_k=10)
```

## Configuration notes

- `threshold` must be a finite number in `[0.0, 1.0]` (default `0.9`).
- Higher thresholds keep more near-variants; lower thresholds collapse more
  aggressively.
- `top_k=None` keeps every survivor; empty input returns `[]`.
