# AbstractKeywordBoost Guide

![AbstractKeywordBoost demo](../assets/abstract-keyword-boost.gif)

Local retrieval postprocessor for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 pipelines (not a DOI connector).

Inspired by LlamaIndex/Haystack keyword boost postprocessors. Soft-boosts hits when abstract metadata or chunk text contains query keywords (from the query string, explicit ``query_terms``, or ``metadata["keywords"]``).

## Usage

```python
from retrieval.abstract_keyword_boost import AbstractKeywordBoost
```

See unit tests for edge cases.
