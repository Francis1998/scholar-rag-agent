# Retracted Filter Guide

![Retracted Filter](../assets/retracted_filter.gif)

Inspired by Retraction Watch / OpenAlex retraction flags used in scholarly search stacks; drops or soft-demotes retracted works via local metadata (not a DOI connector).

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 answer stacks.

## Usage

```python
from retrieval.retracted_filter import RetractedFilter

processor = RetractedFilter()
# call .filter(...) on hybrid SearchResult lists
```

## Safety

Local metadata only — no network calls and no DOI connector side effects.
