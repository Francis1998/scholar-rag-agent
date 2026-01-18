# Performance Tuning For Ingestion

*scholar-rag-agent — 2026-01-18*

## Overview

This guide covers performance tuning for ingestion for the `scholar-rag-agent` project.

## Prerequisites

- Python 3.10+
- Redis (if using distributed mode)
- Environment variables configured (see `.env.example`)

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env

# Run the agent module
python -m agent --help
```

## Common Scenarios

### Scenario 1: Basic Ingestion Usage

```python
from agent import Ingestion

client = Ingestion(config)
result = client.run()
print(result)
```

### Scenario 2: Advanced Configuration

```python
from agent.config import Settings

settings = Settings(
    max_retries=3,
    timeout=30,
    log_level="INFO",
)
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ConnectionError` | API endpoint unreachable | Check `BASE_URL` in `.env` |
| `TimeoutError` | Request took too long | Increase `timeout` setting |
| `AuthError` | Invalid or expired token | Rotate API key |

## See Also

- [README](../README.md)
- [ARCHITECTURE](../ARCHITECTURE.md)
- [API Reference](./API.md)
