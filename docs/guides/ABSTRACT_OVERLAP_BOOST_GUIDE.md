# Abstract Overlap Boost Guide

![Abstract Overlap Boost](../assets/abstract_overlap_boost.gif)

Inspired by LlamaIndex/Haystack keyword-overlap postprocessors; boosts hits whose abstract/summary metadata overlaps the query tokens (not a DOI connector).

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 answer stacks.

## Usage

```python
from retrieval.abstract_overlap_boost import AbstractOverlapBooster

processor = AbstractOverlapBooster()
# call .boost(...) on hybrid SearchResult lists
```

## Safety

Local metadata only — no network calls and no DOI connector side effects.
