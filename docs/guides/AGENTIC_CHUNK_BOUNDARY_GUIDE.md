# Agentic Chunk Boundary Guide

![Agentic chunk boundary demo](../assets/agentic_chunk_boundary.gif)

`AgenticChunkBoundarySplitter` splits long text into agent-ready retrieval
chunks while preferring semantic boundaries over arbitrary character cuts.
It is inspired by LlamaIndex's sentence-window parsing, Chonkie's recursive
chunkers, and LangChain's `MarkdownHeaderTextSplitter`. Despite the
"agentic" name, it requires no LLM or network call: splitting is pure and
deterministic, which is what makes the resulting chunks safe inputs for
downstream agent tool calls and retrieval indexing feeding **GPT-5.5**,
**Claude Sonnet 4.6**, **Gemini 3.x**, or **Kimi K2**.

## How it differs from `TextChunker`

- `TextChunker` (`ingestion/chunking.py`) collapses all whitespace and slices
  a document into fixed-size, overlapping windows. It is simple and fast but
  indifferent to structure: a chunk boundary can land mid-sentence or split a
  heading from its section.
- `AgenticChunkBoundarySplitter` tries the most semantic boundary first and
  only descends to a coarser one when a unit still exceeds `max_chars`. It
  never overlaps chunks, and small heading sections are kept intact rather
  than force-split. A `min_chars` floor additionally reunites undersized
  neighboring chunks so retrieval does not index tiny fragments.

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
6. **Small-chunk merging** runs last: any chunk under `min_chars` is folded
   into a neighbor (the next chunk first, otherwise the previous one) when
   the merge still fits under `max_chars`. This mainly reunites short
   heading sections (like a lone `## Acknowledgments`) with adjacent
   content instead of leaving them as tiny standalone chunks.

```text
headings -> paragraphs -> sentences -> words -> characters -> merge-small
```

## Example

```python
from retrieval.agentic_chunk_boundary import AgenticChunkBoundarySplitter

splitter = AgenticChunkBoundarySplitter(max_chars=1200, min_chars=200)
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
- No chunk ever exceeds `max_chars`. Splitting always recurses into a finer
  boundary (or the character-level fallback) before packing, and the final
  small-chunk merge pass only combines two chunks when the result still
  fits under `max_chars`.
- `min_chars` is best-effort: a chunk that cannot merge with either neighbor
  without exceeding `max_chars` (or that has no eligible neighbor at all)
  may still be returned shorter than `min_chars`.
- `max_chars` must be a positive integer, `min_chars` must be a non-negative
  integer strictly smaller than `max_chars`; invalid values raise
  `ValueError` at construction time. Defaults are `max_chars=1200` and
  `min_chars=200`.
- Blank or whitespace-only input returns an empty list.
