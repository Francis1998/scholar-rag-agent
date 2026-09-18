# Performance Tuning

Scholar RAG Agent is local-first. The default path favors deterministic behavior and
small-corpus ergonomics over distributed throughput.

## Retrieval

- Keep `SCHOLAR_RAG_MAX_SOURCE_DOCS` at or below the default `50` for interactive use; the current executor counts chunk results, not distinct papers.
- Keep multi-hop retrieval at depth `3` for research questions; higher values increase graph fan-out.
- The API uses lexical reranking. A cross-encoder requires explicit Python wiring with `AdaptiveReranker(use_cross_encoder=True)`, its optional dependency, and access to model weights; installing extras alone does not enable it.

## Ingestion

- Chunk size and overlap are controlled in `ingestion.chunking.TextChunker`.
- Larger chunks improve context continuity but reduce retrieval precision.
- Smaller chunks improve citation granularity but increase index size.

## LLM Providers

- API generation uses the reasoning route; speed/cost preferences apply only to callers that request those task types. See the [provider model guide](guides/PROVIDER_MODELS_GUIDE.md).
- The fake adapter demonstrates the workflow, not drafting quality. Measure provider latency and output quality on your own permitted workload rather than treating routing labels as benchmarks.
- Keep provider rate limits conservative for batch corpus analysis.

## Storage

SQLite is adequate for local and small-team workflows. If write contention becomes the bottleneck, keep the storage interfaces and move document/vector/graph persistence behind a dedicated service.
