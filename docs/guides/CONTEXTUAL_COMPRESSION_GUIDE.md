# Contextual Compression Guide

![Contextual compression demo](../assets/contextual_compression.gif)

Use `ContextualCompressor` after retrieval to keep query-relevant evidence while
removing unrelated sentences from each chunk. Compression is deterministic,
dependency-free, and local. Optional LLM stages elsewhere in the stack can use
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 after the bounded context is
assembled.

## How extraction works

For each `SearchResult`, the compressor:

1. Splits chunk text at sentence-ending punctuation and line boundaries.
2. Tokenizes the query and sentences with the shared retrieval tokenizer,
   excluding common stopwords.
3. Keeps sentences that meet the distinct-term `min_overlap`.
4. Selects spans by overlap, then lexical density, with stable source-order ties.
5. Restores selected spans to document order and enforces a hard character cap.

Chunks with no qualifying sentence are filtered out. Scores are preserved, the
new retriever is `contextual_compression`, and prior provenance is appended to
`path`.

## Example

```python
from retrieval.contextual_compression import ContextualCompressor

compressor = ContextualCompressor(
    max_sentences_per_chunk=3,
    max_chars_per_chunk=1_200,
    min_overlap=1,
)
context_results = compressor.compress(
    "How do graph neural networks predict molecular properties?",
    retrieved_results,
    top_k=8,
)
```

## Bounds and behavior

- All constructor bounds must be positive integers.
- `top_k` limits the number of relevant results returned; zero or negative
  values return an empty list.
- Blank or stopword-only queries return an empty list because they carry no
  lexical evidence signal.
- Long selected text is truncated to `max_chars_per_chunk`.
- Input chunks and results are never mutated.

This stage is intended for extractive context reduction, not abstractive
summarization. It therefore adds no network dependency and cannot invent text.
