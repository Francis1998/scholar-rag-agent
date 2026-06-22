# Configuration Reference

Scholar RAG Agent is configured through environment variables loaded by
`pydantic-settings`. Copy `.env.example` to `.env` for local development.

## Required Runtime

- Python 3.11+
- SQLite, provided by the Python standard library
- Optional provider API keys for live LLM or paper metadata calls

## Core Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `SCHOLAR_RAG_DATABASE_PATH` | `.scholar-rag-agent.sqlite3` | SQLite event, document, and graph store. |
| `SCHOLAR_RAG_AGENT_ID` | `local-agent` | Agent ID persisted with event-log entries. |
| `SCHOLAR_RAG_RETRIEVAL_TIMEOUT_SECONDS` | `30` | Retrieval phase timeout. |
| `SCHOLAR_RAG_REASONING_TIMEOUT_SECONDS` | `60` | Reasoning/generation timeout. |
| `SCHOLAR_RAG_MAX_SOURCE_DOCS` | `50` | Maximum source documents per request. |
| `SCHOLAR_RAG_MAX_HOPS` | `5` | Hard graph traversal bound. |
| `SCHOLAR_RAG_DEFAULT_MODEL` | `openai` | Default model family for live adapter routing. |

## Optional Provider Keys

| Variable | Provider |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI GPT-5.5 adapter |
| `ANTHROPIC_API_KEY` | Anthropic Claude Sonnet 4.6 adapter |
| `GEMINI_API_KEY` | Google Gemini 3.1 Pro adapter |
| `MOONSHOT_API_KEY` | Moonshot Kimi K2 adapter |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API connector |

Without these keys, tests and demos use deterministic local fakes.

## Local Commands

```bash
uv sync --extra dev
uv run python scripts/demo_local.py
uv run uvicorn api.main:app --reload
```

## See Also

- [README](../README.md)
- [Architecture](../ARCHITECTURE.md)
- [Safety](../SAFETY.md)
