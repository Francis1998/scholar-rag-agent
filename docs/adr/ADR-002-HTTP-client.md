# ADR-002: Http Client for scholar-rag-agent

**Date:** 2024-03-26
**Status:** Accepted
**Context:** Scientific Rag

## Context

The `agent` module needs a reliable HTTP client solution
that integrates cleanly with our async embedding pipeline.

## Decision

Use **httpx (async)** for HTTP client.

## Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **httpx (async)** (chosen) | Native async, well-maintained | Slightly higher cold-start |
| aiohttp | Mature ecosystem | Sync-first, harder to integrate |
| requests | Zero dependencies | Limited features for production |

## Consequences

- All new embedding components will use `httpx (async)` as the HTTP client layer.
- Existing code will be migrated incrementally.
- Added to `pyproject.toml` as a core dependency.
