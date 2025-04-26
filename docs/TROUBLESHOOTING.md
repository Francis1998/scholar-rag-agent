# Troubleshooting Retrieval

*scholar-rag-agent — 2025-04-26*

## Overview

This guide covers troubleshooting retrieval for the `scholar-rag-agent` project.

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

### Scenario 1: Basic Retrieval Usage

```python
from agent import Retrieval

client = Retrieval(config)
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
