# Scholar RAG Agent

[![CI](https://github.com/Francis1998/scholar-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Francis1998/scholar-rag-agent/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-pytest--cov-blue)](tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Scholar RAG Agent is a production-grade, local-first Agentic RAG system for scientific literature. It ingests papers from PDFs, arXiv, and Semantic Scholar; builds hybrid dense, sparse, and entity-relationship retrieval indexes; and answers research questions with multi-hop reasoning and citation-backed evidence.

The project is designed for the scientific knowledge synthesis narrative behind NIW-style research impact: researchers can accelerate literature review, hypothesis validation, and grounded comparison across large corpora without losing provenance.

```text
                 +---------------------------+
                 | Observe: Query Analyzer   |
                 +-------------+-------------+
                               |
                               v
+---------+      +-------------+-------------+      +-------------------+
| Papers  +----->| Decide: Planner           +----->| Act: Executor     |
+---------+      +-------------+-------------+      +---------+---------+
 PDF/arXiv/S2                  |                              |
                               v                              v
                   +-----------+-----------+       +----------+----------+
                   | SQLite Durable Events |       | Hybrid Retrieval    |
                   +-----------------------+       | Dense + BM25 + RRF  |
                                                   +----------+----------+
                                                              |
                                                              v
                                                   +----------+----------+
                                                   | GraphRAG Multi-hop  |
                                                   +----------+----------+
                                                              |
                                                              v
                                                   +----------+----------+
                                                   | LLM Router + Guard  |
                                                   +----------+----------+
                                                              |
                                                              v
                                                   Citation-backed answer
```

## Install In 3 Commands

```bash
git clone https://github.com/Francis1998/scholar-rag-agent.git
cd scholar-rag-agent && uv sync --extra dev
uv run pytest tests/ -v
```

## Local Demo

```bash
uv run python scripts/demo_local.py
uv run uvicorn api.main:app --reload
```

The deterministic demo ingests a small fixture paper, executes an Observe -> Decide -> Act run, prints the planner trace, and returns a cited answer. A generated demo asset is available at `docs/assets/demo.gif`.

## Provider Keys

All live providers are optional. Without keys the system uses deterministic fakes for tests and demos. Configure keys in `.env` or your shell:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export MOONSHOT_API_KEY=...
```

## Quality Gates

```bash
uv run ruff check src/
uv run mypy src/
uv run pytest tests/ -v
```

## License

Apache-2.0. See `LICENSE`.
