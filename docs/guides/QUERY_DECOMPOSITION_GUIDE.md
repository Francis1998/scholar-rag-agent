# Query Decomposition Guide

![Query decomposition demo](../assets/query_decomposition.gif)

`QueryDecomposer` turns a compound research question into multiple retrieval
sub-queries. Inspired by LlamaIndex and Haystack multi-query patterns, the
splitter is local and deterministic: it uses conjunctions and question marks
rather than an LLM. The original query is always preserved as the first item so
fusion stages can keep a full-question baseline. Downstream synthesis can still
use GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or Kimi K2.

## Split heuristics

The decomposer:

1. Normalizes whitespace.
2. Keeps the full original query as part index `0`.
3. Splits on conjunction phrases such as `and`, `as well as`, `along with`,
   `in addition to`, `and also`, and `and then`.
4. Also splits on `;` and `?` boundaries for multi-sentence questions.
5. Deduplicates parts case-insensitively while preserving first-seen order.
6. Optionally truncates the list with `max_parts`.

Blank or punctuation-only inputs return an empty list. `max_parts`, when set,
must be a positive integer.

## Example

```python
from retrieval.query_decomposition import QueryDecomposer

parts = QueryDecomposer().decompose(
    "graph neural networks for molecules and sparse retrieval for ranking",
    max_parts=5,
)
# [
#   "graph neural networks for molecules and sparse retrieval for ranking",
#   "graph neural networks for molecules",
#   "sparse retrieval for ranking",
# ]

result_lists = [await retriever.retrieve(part) for part in parts]
fused = reciprocal_rank_fusion(result_lists)
```

Use the original query as the primary retrieval path and the remaining parts as
focused expansions. This complements `QueryRewriter` synonym expansion and
Multi-HyDE hypothetical abstracts.
