# Author Count Boost Guide

![Author Count Boost](../assets/author_count_boost.gif)

Inspired by bibliometric priors in scholarly RAG (Haystack-style metadata boosts); softly prefers mid-sized author lists over single-author or extreme mega-author rows (not a DOI connector).

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 answer stacks.

## Usage

```python
from retrieval.author_count_boost import AuthorCountBooster

processor = AuthorCountBooster()
# call .boost(...) on hybrid SearchResult lists
```

## Safety

Local metadata only — no network calls and no DOI connector side effects.
