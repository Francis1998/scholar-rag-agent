# Agentic Chunk Boundary Guide

![Agentic chunk boundary demo](../assets/agentic_chunk_boundary.gif)

`AgenticChunkBoundarySplitter` splits long text into agent-ready retrieval
chunks while preferring semantic boundaries over arbitrary character cuts.
It is inspired by LlamaIndex's sentence-window parsing, Chonkie's recursive
chunkers, and LangChain's `MarkdownHeaderTextSplitter`. Despite the
"agentic" name, it requires no LLM or network call: splitting is pure and
deterministic, which is what makes the resulting chunks safe inputs for
downstream agent tool calls and retrieval indexing.

## How it differs from `TextChunker`

- `TextChunker` (`ingestion/chunking.py`) collapses all whitespace and slices
  a document into fixed-size, overlapping windows. It is simple and fast but
  indifferent to structure: a chunk boundary can land mid-sentence or split a
  heading from its section.
- `AgenticChunkBoundarySplitter` tries the most semantic boundary first and
  only descends to a coarser one when a unit still exceeds `max_chars`. It
  never overlaps chunks, and small heading sections are kept intact rather
  than force-split.

## Boundary priority

1. **Markdown headings** (`#` through `######`) anchor section boundaries.
   A section that already fits in `max_chars`, heading included, is kept
   whole.
2. **Blank-line paragraph breaks** split an oversized section. Paragraphs are
   greedily packed back together (joined with `\n\n`) up to `max_chars`.
3. **Sentence boundaries** (`.`, `!`, `?`) split an oversized paragraph.
   Sentences are greedily packed (joined with a space) up to `max_chars`.
4. **Word boundaries** split an oversized sentence that has no further
   sentence breaks.
5. **Raw character slicing** is the last resort, used only for a single
   token that alone exceeds `max_chars` (e.g. a long identifier or URL).

```text
headings -> paragraphs -> sentences -> words -> characters
```

## Example

```python
from ingestion.agentic_chunk_boundary import AgenticChunkBoundarySplitter

splitter = AgenticChunkBoundarySplitter(max_chars=800)
chunks = splitter.split(markdown_text)
```

Use it as a drop-in alternative to `TextChunker` when ingesting a `Document`:

```python
from retrieval.models import Document

document = Document(document_id="d1", title="Paper", text=markdown_text, source="pdf")
retrieval_chunks = splitter.chunk(document)
```

`chunk()` returns the same `Chunk` model as `TextChunker`, with deterministic
ids derived from the document id, chunk index, and chunk text, and a
`chunk_index` metadata field for stable ordering.

## Normalization and safety

- Paragraph and sentence text has internal whitespace (including line wraps)
  collapsed to single spaces before packing; markdown headings and blank
  lines between paragraphs are preserved as boundaries.
- No chunk exceeds `max_chars`, except that this is only guaranteed once
  splitting reaches the character-level fallback; every higher level
  recurses into a finer boundary before falling back further.
- `max_chars` must be a positive integer; invalid values raise `ValueError`
  at construction time.
- Blank or whitespace-only input returns an empty list.
