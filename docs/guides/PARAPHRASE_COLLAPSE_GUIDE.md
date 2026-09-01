# ParaphraseCollapser Guide

![demo](../assets/paraphrase_collapse.gif)

Gap fill vs LlamaIndex/Haystack paraphrase-aware dedupe stages that word-Jaccard near-duplicate collapse misses.

Works with GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 scholarly RAG pipelines.
Not a DOI connector.

## Usage

```python
from retrieval.paraphrase_collapse import ParaphraseCollapser

stage = ParaphraseCollapser()
# call stage.collapse(...)
```

See unit tests for edge cases.
