# Sentence Window Expand Guide

![Sentence window expand demo](../assets/sentence_window_expand.gif)

`SentenceWindowExpander` widens each retrieved chunk with neighboring
sentences from the full document text stored in chunk metadata. It is inspired
by LlamaIndex `SentenceWindowNodeParser` / sentence-window retrieval, but is
fully local and deterministic. Expanded evidence can then feed GPT-5.5,
Claude Sonnet 4.6, Gemini 3.x, or Kimi K2 for grounded synthesis.

## How it differs from related stages

- `ParentDocumentExpander` replaces a child hit with an entire parent document
  from an external in-memory store keyed by `document_id`.
- `ContextualCompressor` *shrinks* chunk text to query-relevant sentence spans.
- `SentenceWindowExpander` *adds* ±`window_sentences` neighbors around the
  retrieved span when `document_text` or `full_text` metadata is present.

## How expansion works

1. Read full document text from `chunk.metadata["document_text"]`, falling
   back to `chunk.metadata["full_text"]`.
2. If neither key is set, or the chunk text cannot be located inside the
   document, return a deep copy of the result with the original chunk text.
3. Split the document on `.` / `!` / `?` or newlines into sentences.
4. Locate the sentence span covering the retrieved chunk text.
5. Include sentences from `start - window_sentences` through
   `end + window_sentences`, clamped to document bounds.
6. Return a new `SearchResult` / `Chunk` copy (inputs are never mutated).

When the text actually expands, `retriever` becomes `sentence_window` and the
prior retriever is appended to `path`. Unchanged copies preserve the original
`retriever` and `path`.

## Example

```python
from retrieval.sentence_window_expand import SentenceWindowExpander

expander = SentenceWindowExpander(window_sentences=1)
expanded = expander.expand(retrieved_results)
```

Store the full paper (or section) text on each child chunk at ingest time:

```python
chunk.metadata["document_text"] = full_paper_text
```

## Configuration notes

- `window_sentences` must be an integer in `0..10` (default `1`).
- `window_sentences=0` never adds neighbors; the original chunk text is kept.
- Boolean values are rejected even though `bool` subclasses `int`.
- Empty input returns an empty list of copies is not applicable — `expand([])`
  returns `[]`.
