# scholar-rag-agent

![License](https://img.shields.io/badge/license-MIT-green) ![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen) ![CI](https://github.com/Francis1998/{repo}/actions/workflows/ci.yml/badge.svg)

> Scientific Rag — powered by modern Python async architecture.

## Features

- **Retrieval engine** with configurable strategies
- **Ingestion pipeline** with full observability
- **Async-first** design using `asyncio` + `aiohttp`
- **Type-safe** with full `mypy` compliance
- **Production-ready** with Docker, CI/CD, and structured logging

## Quick Start

```bash
git clone https://github.com/Francis1998/scholar-rag-agent.git
cd scholar-rag-agent
pip install -e ".[dev]"
cp .env.example .env
python -m agent --help
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | System design and component overview |
| [Configuration](docs/CONFIGURATION.md) | All configuration options |
| [Deployment](docs/DEPLOYMENT.md) | Production deployment guide |
| [Contributing](CONTRIBUTING.md) | Development and PR workflow |
| [Changelog](CHANGELOG.md) | Version history |

## License

MIT © [Francis1998](https://github.com/Francis1998)

*Last updated: 2026-02-08*
