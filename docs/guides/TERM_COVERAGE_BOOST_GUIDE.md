# TermCoverageBooster Guide

![demo](../assets/term_coverage_boost.gif)

Gap fill vs Elasticsearch/Haystack coverage boosts distinct from symmetric Jaccard lexical overlap.

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 scholarly RAG pipelines.
Not a DOI connector.

## Usage

```python
from retrieval.term_coverage_boost import TermCoverageBooster

stage = TermCoverageBooster()
# call stage.boost(...)
```

See unit tests for edge cases.
