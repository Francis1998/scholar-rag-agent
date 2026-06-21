# Safety Controls

## Timeout Policy

Retrieval defaults to 30 seconds and reasoning defaults to 60 seconds. Both are configurable through environment variables and enforced around async execution.

## Scope Bounds

The default source cap is 50 documents per query. Multi-hop graph traversal defaults to depth 3 and is globally bounded by `SCHOLAR_RAG_MAX_HOPS`, never exceeding 5.

## Cancellation

`CancellationToken` is checked at every state transition and inside retrieval loops. Cancelled runs transition to `ERROR` with a structured payload.

## Hallucination Guard

Generated answers must include claims mapped to source chunk IDs. Claims without supporting retrieved chunks are marked `[UNGROUNDED]`, and the response includes a warning.

## Provider Backoff

LLM calls pass through per-provider rate limiters with exponential backoff for transient `429`, `500`, `502`, `503`, and `504` failures.
