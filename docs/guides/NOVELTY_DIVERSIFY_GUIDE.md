# Novelty Diversify Guide

![Novelty Diversify](../assets/novelty_diversify.gif)

Inspired by LlamaIndex diversity postprocessors and Maximal Marginal Relevance
(MMR); greedily re-ranks hits while soft-demoting near-duplicate chunk text via
token Jaccard overlap (not a DOI connector).

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 answer stacks.

## How it differs from peers

| Stage | Behavior |
| --- | --- |
| `MMRDiversifier` | Reorders with λ·relevance − (1−λ)·similarity; scores unchanged |
| `NearDuplicateCollapser` | Hard-drops near-duplicates above a threshold |
| `NoveltyDiversifier` | Greedy selection with alpha-blended novelty scores (keeps all up to `top_k`) |

## Scoring

At each greedy step, for every remaining candidate `d`:

```text
novelty(d) = 1 - max Jaccard(tokens(d), tokens(s)) over already-selected s
             (novelty = 1.0 when nothing is selected yet)
new_score  = (1 - alpha) * old_score + alpha * novelty(d)
```

The highest `new_score` candidate is appended, then the loop repeats until
`top_k` (or all) rows are chosen. Inputs are not mutated; returned rows use
`retriever="novelty_diversify"`.

## Usage

```python
from retrieval.novelty_diversify import NoveltyDiversifier

processor = NoveltyDiversifier(alpha=0.5)
diversified = processor.diversify(retrieved_results, top_k=10)
```

## Safety

Local chunk text only — no network calls and no DOI connector side effects.
