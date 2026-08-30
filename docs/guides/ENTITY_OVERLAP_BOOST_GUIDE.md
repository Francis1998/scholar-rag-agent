# Entity Overlap Boost Guide

![Entity Overlap Boost](../assets/entity_overlap_boost.gif)

Inspired by entity-centric retrieval in GraphRAG/Haystack; boosts hits whose chunk text shares capitalized entity tokens with the query (not a DOI connector).

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 answer stacks.

## Usage

```python
from retrieval.entity_overlap_boost import EntityOverlapBooster

processor = EntityOverlapBooster()
# call .boost(...) on hybrid SearchResult lists
```

## Safety

Local chunk text only — no network calls and no DOI connector side effects.
