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

`max_chunks_per_document` is an optional **request field**, not an environment
setting. `/query`, `/retrieve`, and their runner methods accept strict integers
1-50; HTTP null is invalid and omission leaves the existing behavior unchanged.
It limits passages per document after reranking without increasing
`SCHOLAR_RAG_MAX_SOURCE_DOCS`. Policy is saved separately from the unchanged
four-field run configuration. See
[Per-paper evidence limits](docs/guides/PER_PAPER_EVIDENCE_LIMITS_GUIDE.md).

`min_evidence_documents` is also **request-only**, not an environment setting.
`/query`, `/retrieve`, and their runner methods accept strict integers 1-50;
HTTP null is invalid, while omission or Python `None` leaves it unconfigured.
It checks distinct document IDs in final captured context after reranking and
quotas, without increasing any limits. Unmet minima stop answer generation;
previews retain evidence and count diagnostics. See
[Minimum evidence documents](docs/guides/MINIMUM_EVIDENCE_DOCUMENTS_GUIDE.md).

Saved bibliography exports add no setting or provider requirement. They read
completed frozen events only, enforce fixed 50-source and 256-KiB-per-format
bounds, and do not use today's model configuration or corpus to enrich fields.
See [the BibTeX/JSON contract](docs/guides/SAVED_BIBLIOGRAPHY_GUIDE.md).

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
source-linked defaults rechecked on **2026-09-29 America/Los_Angeles** and payload
compatibility limits. This is documentation verification, not a live inference
test or confirmation of account entitlement.

Gemini still uses `generateContent`, not the current Interactions quickstart.
Google rejects unrestricted standard keys; auth keys and appropriately restricted
standard keys are covered in the [Gemini compatibility notes](docs/guides/PROVIDER_MODELS_GUIDE.md#google-gemini).
Setting an API key header does not migrate the endpoint or the key's permissions.

Live adapters reject invalid JSON, absent/blank final answer text, and explicit
output/context-limit truncation with a nonretryable provider-response error.
`/query` records an `ERROR` run rather than a completed empty or known-partial
answer. Existing payload budgets and routing are unchanged; see
[finish-reason handling](docs/guides/PROVIDER_MODELS_GUIDE.md#explicitly-truncated-answers).

Live rate admission counts every HTTP attempt, including failed requests and
retries, against the default 60 attempts per adapter instance per sliding minute.
Retries wait for capacity after the existing backoff; the three-retry cap and
phase timeouts are unchanged. This is not a shared account-wide quota. See
[provider backoff and rate limits](SAFETY.md#provider-backoff).

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the extended reference and local commands.
