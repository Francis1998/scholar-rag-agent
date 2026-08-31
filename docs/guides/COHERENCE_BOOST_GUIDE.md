# Coherence Boost Guide

![Coherence Boost](../assets/coherence_boost.gif)

Inspired by LlamaIndex/Haystack coherence-aware rerankers; boosts hits whose
chunk text shows adjacent-sentence token overlap and consistent query-term
density across sentences (not a DOI connector).

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 answer stacks.

## How scoring works

1. Split `chunk.text` on sentence boundaries.
2. **Neighbor overlap** — mean Jaccard similarity of token sets between
   adjacent sentences (`0.0` when fewer than two sentences).
3. **Query continuity** — fraction of sentences containing at least one query
   token (`0.0` when the query has no tokens).
4. `coherence = 0.5 * neighbor_overlap + 0.5 * query_continuity`

```text
new_score = (1 - alpha) * old_score + alpha * coherence
```

Results are sorted by `new_score` descending (stable). Input `SearchResult`
objects are not mutated; returned rows use `retriever="coherence_boost"`.

## Usage

```python
from retrieval.coherence_boost import CoherenceBooster

processor = CoherenceBooster(alpha=0.3)
boosted = processor.boost(retrieved_results, query="transformer attention", top_k=10)
```

## Safety

Local chunk text and query tokens only — no network calls and no DOI connector
side effects.
