# ADR-075: Async Task Queue for scholar-rag-agent

**Date:** 2024-03-11
**Status:** Accepted
**Context:** Scientific Rag

## Context

The `agent` module needs a reliable async task queue solution
that integrates cleanly with our async graph pipeline.

## Decision

Use **Redis Streams** for async task queue.

## Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **Redis Streams** (chosen) | Native async, well-maintained | Slightly higher cold-start |
| Celery + RabbitMQ | Mature ecosystem | Sync-first, harder to integrate |
| asyncio.Queue | Zero dependencies | Limited features for production |

## Consequences

- All new graph components will use `Redis Streams` as the async task queue layer.
- Existing code will be migrated incrementally.
- Added to `pyproject.toml` as a core dependency.
