# Configuration

Scholar RAG Agent uses `pydantic-settings` and environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SCHOLAR_RAG_DATABASE_PATH` | `.scholar-rag-agent.sqlite3` | SQLite event/document/graph store path. |
| `SCHOLAR_RAG_AGENT_ID` | `local-agent` | Agent identifier persisted in events. |
| `SCHOLAR_RAG_RETRIEVAL_TIMEOUT_SECONDS` | `30` | Retrieval phase timeout. |
| `SCHOLAR_RAG_REASONING_TIMEOUT_SECONDS` | `60` | Reasoning/generation timeout. |
| `SCHOLAR_RAG_MAX_SOURCE_DOCS` | `50` | Maximum source documents per query. |
| `SCHOLAR_RAG_MAX_HOPS` | `5` | Global hop bound, with default retrieval depth set to 3. |
| `SCHOLAR_RAG_DEFAULT_MODEL` | `openai` | Default model family for live adapter routing. |

Provider keys are optional: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MOONSHOT_API_KEY`, and `SEMANTIC_SCHOLAR_API_KEY`.

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the extended reference and local commands.
