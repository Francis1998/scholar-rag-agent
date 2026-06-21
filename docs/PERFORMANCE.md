# Performance Tuning

Scholar RAG Agent is local-first. The default path favors deterministic behavior and
small-corpus ergonomics over distributed throughput.

## Retrieval

- Keep `SCHOLAR_RAG_MAX_SOURCE_DOCS` at or below the default `50` for interactive use.
- Keep multi-hop retrieval at depth `3` for research questions; higher values increase graph fan-out.
- Use the lexical reranker for fast local demos and enable cross-encoder reranking only when the model dependency is installed and latency is acceptable.

## Ingestion

- Chunk size and overlap are controlled in `ingestion.chunking.TextChunker`.
- Larger chunks improve context continuity but reduce retrieval precision.
- Smaller chunks improve citation granularity but increase index size.

## LLM Providers

- Route fast drafting tasks to Gemini or the fake adapter.
- Route deeper reasoning tasks to Claude/OpenAI when API keys are configured.
- Keep provider rate limits conservative for batch corpus analysis.

## Storage

SQLite is adequate for local and small-team workflows. If write contention becomes the bottleneck, keep the storage interfaces and move document/vector/graph persistence behind a dedicated service.
