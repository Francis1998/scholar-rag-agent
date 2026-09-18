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
| `SCHOLAR_RAG_DEFAULT_MODEL` | `openai` | Provider family for DEFAULT tasks and missing-provider fallbacks, not an API model ID. |
| `SCHOLAR_RAG_OPENAI_MODEL` | `gpt-6-astra` | OpenAI Chat Completions model ID. |
| `SCHOLAR_RAG_ANTHROPIC_MODEL` | `claude-sonnet-5` | Anthropic Messages model ID. |
| `SCHOLAR_RAG_GEMINI_MODEL` | `gemini-3.8-flash` | Gemini generateContent model ID. |
| `SCHOLAR_RAG_KIMI_MODEL` | `kimi-k3` | Moonshot Chat Completions model ID. |
| `PdfOcrHook.min_chars` | `40` | Constructor threshold: stripped pypdf text shorter than this triggers `OcrBackend` (default `NullOcrBackend`). |

Provider keys are optional: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MOONSHOT_API_KEY`, and `SEMANTIC_SCHOLAR_API_KEY`.

Model IDs are stripped of surrounding whitespace and must be non-empty strings.
Unset values use the defaults; explicit blank values fail settings validation,
even without API keys. Model IDs alone never enable live calls.

API synthesis requests use REASONING routing, preferring Anthropic when its key
is configured. SPEED prefers Gemini, COST prefers Kimi, and DEFAULT uses
`SCHOLAR_RAG_DEFAULT_MODEL`. A missing preferred provider falls back to the
configured default family, then OpenAI, then the offline fake. This is
configuration fallback, not failover after an HTTP error.

See the [provider model guide](docs/guides/PROVIDER_MODELS_GUIDE.md) for
source-linked defaults checked on 2026-09-17 and payload compatibility limits.
See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the extended reference and local commands.
