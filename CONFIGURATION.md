# Configuration

Scholar RAG Agent uses `pydantic-settings` and environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SCHOLAR_RAG_DATABASE_PATH` | `.scholar-rag-agent.sqlite3` | SQLite event/document/graph store path. |
| `SCHOLAR_RAG_AGENT_ID` | `local-agent` | Agent identifier persisted in events. |
| `SCHOLAR_RAG_RETRIEVAL_TIMEOUT_SECONDS` | `30` | Finite retrieval phase timeout in seconds, at least `1`. |
| `SCHOLAR_RAG_REASONING_TIMEOUT_SECONDS` | `60` | Finite reasoning/generation timeout in seconds, at least `1`. |
| `SCHOLAR_RAG_MAX_SOURCE_DOCS` | `50` | Maximum retrieved chunk results per query, despite the historical setting name; not a distinct-document quota. |
| `SCHOLAR_RAG_MAX_HOPS` | `5` | Global hop bound; planner tasks request one to three hops depending on intent. |
| `SCHOLAR_RAG_DEFAULT_MODEL` | `openai` | Provider family for DEFAULT tasks and missing-provider fallbacks, not an API model ID. |
| `SCHOLAR_RAG_OPENAI_MODEL` | `gpt-6-astra` | OpenAI Chat Completions model ID. |
| `SCHOLAR_RAG_ANTHROPIC_MODEL` | `claude-sonnet-5` | Anthropic Messages model ID. |
| `SCHOLAR_RAG_GEMINI_MODEL` | `gemini-3.8-flash` | Gemini generateContent model ID. |
| `SCHOLAR_RAG_KIMI_MODEL` | `kimi-k3` | Moonshot Chat Completions model ID. |
| `PdfOcrHook.min_chars` | `40` | Constructor threshold: stripped pypdf text shorter than this triggers `OcrBackend` (default `NullOcrBackend`). |

Provider keys are optional: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MOONSHOT_API_KEY`, and `SEMANTIC_SCHOLAR_API_KEY`.

Invalid timeouts, including `NaN`, infinities, and overflow such as `1e999`, fail
settings validation at startup; they are not clamped or replaced with defaults.
Python `SafetyLimits` and effective `RunConfiguration` accept finite positive
subsecond values such as `0.1`; no additional maximum is imposed. See the
[timeout policy](SAFETY.md#timeout-policy) for coercion and runtime error behavior.

Model IDs are stripped of surrounding whitespace and must be non-empty strings.
Unset values use the defaults; explicit blank values fail settings validation,
even without API keys. Model IDs alone never enable live calls.

API synthesis requests use REASONING routing, preferring Anthropic when its key
is configured. SPEED prefers Gemini, COST prefers Kimi, and DEFAULT uses
`SCHOLAR_RAG_DEFAULT_MODEL`. A missing preferred provider falls back to the
configured default family, then OpenAI, then the offline fake. This is
configuration fallback, not failover after an HTTP error.

See the [provider model guide](docs/guides/PROVIDER_MODELS_GUIDE.md) for
source-linked defaults rechecked on **2026-09-26 America/Los_Angeles** and payload
compatibility limits. This is documentation verification, not a live inference
test or confirmation of account entitlement.
Live adapters reject invalid JSON or absent/blank final answer text with an
explicit, nonretryable provider-response error; `/query` records an `ERROR` run
rather than a completed empty answer.

Live rate admission counts every HTTP attempt, including failed requests and
retries, against the default 60 attempts per adapter instance per sliding minute.
Retries wait for capacity after the existing backoff; the three-retry cap and
phase timeouts are unchanged. This is not a shared account-wide quota. See
[provider backoff and rate limits](SAFETY.md#provider-backoff).

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the extended reference and local commands.
