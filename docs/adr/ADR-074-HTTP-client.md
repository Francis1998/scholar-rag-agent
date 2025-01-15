# ADR-074: Http Client for scholar-rag-agent

**Date:** 2025-01-15
**Status:** Accepted
**Context:** Scientific Rag

## Context

The `agent` module needs a reliable HTTP client solution
that integrates cleanly with our async retrieval pipeline.

## Decision

Use **httpx (async)** for HTTP client.

## Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **httpx (async)** (chosen) | Native async, well-maintained | Slightly higher cold-start |
| aiohttp | Mature ecosystem | Sync-first, harder to integrate |
| requests | Zero dependencies | Limited features for production |

## Consequences

- All new retrieval components will use `httpx (async)` as the HTTP client layer.
- Existing code will be migrated incrementally.
- Added to `pyproject.toml` as a core dependency.
