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
| `SCHOLAR_RAG_MAX_SOURCE_DOCS` | `50` | Maximum retrieved chunk results per request, not a distinct-document quota. |
| `SCHOLAR_RAG_MAX_HOPS` | `5` | Hard graph traversal bound. |
| `SCHOLAR_RAG_DEFAULT_MODEL` | `openai` | Provider family for DEFAULT tasks and missing-provider fallbacks, not an API model ID. |
| `PdfOcrHook.min_chars` | `40` | Constructor threshold: stripped pypdf text shorter than this triggers `OcrBackend` (default `NullOcrBackend`). |

## Provider Model IDs

| Environment variable | `Settings` field | Default API model ID |
| --- | --- | --- |
| `SCHOLAR_RAG_OPENAI_MODEL` | `openai_model` | `gpt-6-astra` |
| `SCHOLAR_RAG_ANTHROPIC_MODEL` | `anthropic_model` | `claude-sonnet-5` |
| `SCHOLAR_RAG_GEMINI_MODEL` | `gemini_model` | `gemini-3.8-flash` |
| `SCHOLAR_RAG_KIMI_MODEL` | `kimi_model` | `kimi-k3` |

Settings precedence is explicit `Settings(...)` arguments, process environment,
`.env`, then the built-in defaults. Model IDs are trimmed and must be non-empty
strings; explicit empty, whitespace-only, null, or non-string values fail
validation rather than silently reverting to a default. Leave a variable unset
to use its default. Validation also applies when no provider key is configured.

For example, choose a different available OpenAI model without changing the
provider-routing policy:

```bash
export SCHOLAR_RAG_OPENAI_MODEL=gpt-5.6-terra
```

Or set `Settings(openai_model="gpt-5.6-terra")` in Python. `build_model_router`
passes each field to its corresponding adapter only when that provider has a
key. Direct adapter callers can still pass `model="..."` explicitly; constructors
do not load or override that choice from application environment settings.

These fields accept custom IDs, not a hard-coded catalog allowlist. The selected
model must support the adapter's endpoint and bounded, single-turn text payload.
No model discovery, account-entitlement check, or inference call happens during
settings validation. See the [provider model guide](guides/PROVIDER_MODELS_GUIDE.md)
for the **2026-09-19 America/Los_Angeles** catalog check and model-specific limitations.

## Optional Provider Keys

| Variable | Provider |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI Chat Completions adapter |
| `ANTHROPIC_API_KEY` | Anthropic Messages adapter |
| `GEMINI_API_KEY` | Google Gemini generateContent adapter |
| `MOONSHOT_API_KEY` | Moonshot Kimi Chat Completions adapter |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API connector |

Without LLM provider keys, generation uses deterministic local fakes. Setting a
model ID alone does not enable HTTP calls. `SEMANTIC_SCHOLAR_API_KEY` is for paper
metadata, not LLM routing.

## Routing and Fallbacks

| Request task type | Preferred provider |
| --- | --- |
| `REASONING` | `anthropic` |
| `SPEED` | `gemini` |
| `COST` | `kimi` |
| `DEFAULT` | The family in `SCHOLAR_RAG_DEFAULT_MODEL` |

Provider families are `openai`, `anthropic`, `gemini`, `kimi`, and `fake`. If the
preferred provider has no configured adapter, routing tries the configured
default family, then OpenAI, then the fake. It does not search every keyed
provider. These are static task preferences, not measured speed/cost guarantees.

The FastAPI query path uses `RoutingLLMAdapter`, and synthesis in `Executor`
always requests `REASONING`. Changing `SCHOLAR_RAG_DEFAULT_MODEL` therefore does
not override a configured Anthropic provider for API answers; it chooses the
fallback if Anthropic is absent. Even `SCHOLAR_RAG_DEFAULT_MODEL=fake` does not
force all tasks offline when preferred live providers have keys.

This fallback policy selects providers before a call. HTTP failures are surfaced
after the existing transient-error retries; they do not trigger cross-provider
or fake failover.

## Response Provenance

Live adapters populate `LLMResponse.raw_provider` with the provider family and
`LLMResponse.model_name` with the configured/requested API model ID. The optional
`model_name` defaults to `None` for the fake and existing custom response
producers. It does not identify an immutable server-resolved snapshot. This
normalized LLM response does not include credentials or request headers, and it
is not a new field on the public query response.

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
