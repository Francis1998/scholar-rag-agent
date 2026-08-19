# Parent Document Expansion Guide

![Parent document expansion demo](../assets/parent_document.gif)

Use `ParentDocumentExpander` when retrieval should search small child chunks but
generation needs the surrounding parent document. Expansion is an in-memory,
deterministic lookup with no network or LLM dependency. Optional stages
elsewhere in the stack can send expanded evidence to GPT-5.5 / Claude Sonnet
4.6 / Gemini 3.x / Kimi K2.

## Parent store contract

Pass a mapping keyed by the `document_id` carried by child chunks. Values can be:

- `Document` objects for normalized full documents.
- `Chunk` objects when a parent-sized chunk already exists.
- Strings for lightweight stores containing only full parent text.

The mapping is copied at construction. Unsupported value types fail fast.

## Example

```python
from retrieval.parent_document import ParentDocumentExpander

parents = {
    "paper-1": full_document,
    "paper-2": "Complete text for the second paper.",
}
expander = ParentDocumentExpander(parents)
parent_results = expander.expand(child_chunk_results, top_k=5)
```

For string parents, the child supplies title, source, and metadata provenance.
For `Document` and `Chunk` parents, the stored parent fields are retained.

## Ranking and deduplication

- Child hits without a matching parent id are skipped.
- Multiple children from one parent collapse to one result.
- The highest child score becomes the parent score.
- Parent results are sorted by score, with deterministic first-seen tie order.
- `top_k` bounds parent results; zero or negative values return an empty list.

Expanded results use `retriever="parent_document"` and append the winning child
retriever to `path`. Inputs and stored parent models are not mutated.
