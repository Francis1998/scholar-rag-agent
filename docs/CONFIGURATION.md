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
| `SCHOLAR_RAG_RETRIEVAL_TIMEOUT_SECONDS` | `30` | Finite retrieval phase timeout in seconds, at least `1`. |
| `SCHOLAR_RAG_REASONING_TIMEOUT_SECONDS` | `60` | Finite reasoning/generation timeout in seconds, at least `1`. |
| `SCHOLAR_RAG_MAX_SOURCE_DOCS` | `50` | Maximum retrieved chunk results per request, not a distinct-document quota. |
| `SCHOLAR_RAG_MAX_HOPS` | `5` | Hard graph traversal bound. |
| `SCHOLAR_RAG_DEFAULT_MODEL` | `openai` | Exact lowercase provider family for DEFAULT tasks and missing-provider fallbacks: `openai`, `anthropic`, `gemini`, `kimi`, or `fake`; not an API model ID. |
| `PdfOcrHook.min_chars` | `40` | Constructor threshold: stripped pypdf text shorter than this triggers `OcrBackend` (default `NullOcrBackend`). |

`SCHOLAR_RAG_DEFAULT_MODEL` (`Settings.default_model`) accepts only the exact
lowercase families above, not API model IDs or aliases. Its parsed value is
neither trimmed nor case-normalized. Blank, unknown, differently cased, or
surrounding-whitespace values fail settings validation before the application
container, storage, or providers initialize, including without API keys.
Leave it unset to use `openai`; invalid values do not trigger routing fallback.
Normal `.env` syntax parsing still applies: quotes preserve surrounding spaces,
which are then rejected. Explicit `Settings(...)` arguments override process
environment, then `.env`, then built-in defaults.

Both timeout settings require finite values of at least one second, including
values from process environment and `.env`. `NaN`, infinities, and floating-point
overflow such as `1e999` fail at startup, before storage/provider setup. Invalid
values are never silently clamped or replaced with defaults.

Python `SafetyLimits` and effective `RunConfiguration` instead accept any finite
positive timeout, including `0.1` seconds; there is no additional maximum. They
retain Pydantic float coercion: numeric strings and integers are accepted,
`True` becomes `1.0`, and `False` is rejected as zero. Mutable `SafetyLimits`
are revalidated when copied at run/preview entry, before asynchronous work.
See the [timeout policy](../SAFETY.md#timeout-policy) for error behavior and
cooperative-timeout limitations.

Python `SafetyLimits` count bounds are stricter than timeout coercion:
`max_source_docs` must be an integer at least 1 and `max_hops` an integer at least
0; booleans, floats, strings, and `None` are invalid. Construction and dataclass
replacement reject invalid configuration rather than clamping it. Defaults stay
50/5, and larger Python bounds such as 100/8 are supported. Valid configurations
still clamp negative requested counts to 1 source or 0 hops. API `Settings`,
environment parsing, and evidence schemas are unchanged. See
[scope bounds](../SAFETY.md#scope-bounds) for pre-work errors and copy isolation.

## Per-request evidence quotas

`max_chunks_per_document` is optional on `/query`, `/retrieve`, `AgentRunner.run`,
and `AgentRunner.preview`. It accepts strict integers 1-50, not booleans, strings,
or floats. Omit it for unchanged behavior; HTTP null is rejected. There is no new
environment setting. The frozen policy filters the existing post-rerank pool
without extra retrieval and can leave fewer than `max_source_docs` passages.
It is retained in the plan/events/exports, not added to `RunConfiguration`.
See [the complete guide and offline demo](guides/PER_PAPER_EVIDENCE_LIMITS_GUIDE.md).

`near_duplicate_threshold` opts both endpoints and runner methods into lexical
collapse **after reranking and before quotas/minimum-document assessment**.
It accepts finite strict numbers greater than zero and at most one; bools,
strings, explicit HTTP null, non-finite values, and out-of-range values fail
before work. Omission or Python `None` disables it. No environment setting,
extra retrieval, model, endpoint, or budget change is introduced.
The frozen policy and survivor paths remain reviewable in previews and saved
evidence; threshold `1` means equal meaningful-term sets, not identical text.
See [the complete collapse guide](guides/NEAR_DUPLICATE_COLLAPSE_GUIDE.md).

## Source-context reader bounds

`GET /documents/context`, `/explore/context`, and `SQLiteSourceContext.read`
use the existing corpus database without any new environment setting. The
request supplies exact document/chunk IDs and 0-5 before/after neighbors,
defaulting to two on each side. One snapshot validates at most 2,048 chunks;
metadata over 8,192 stored bytes per chunk fails before JSON parsing.
These bounds are independent of retrieval limits, page size and provider
budgets. Reducing the window does not bypass source-order validation.
Standalone Python reading avoids normal app startup/index initialization.
See [the complete API/Python/browser guide](guides/SOURCE_CONTEXT_GUIDE.md).

## Browser passage-search bounds

`GET /explore/search` uses the existing corpus and `SQLiteLiteralSearch` with
no new environment setting or provider dependency. Phrases are 1-200 Unicode
characters; page size is 1-50 (default 20); exact paper and saved collection
scopes are mutually exclusive. Shared 4-MiB stored-text, 256-KiB serialized
search and five-second scan bounds apply, plus the explorer's 1-MiB escaped
HTML cap. Catalog-return page size remains independently 1-100. Native GET
phrases appear in URLs/history/logs; configure your own access/logging boundary.
See [browser/API/Python linkage and error recovery](guides/BROWSER_PASSAGE_SEARCH_GUIDE.md).

## Saved-run browser bounds

`GET /explore/runs` reuses `SQLiteRunHistory`: default 20, maximum 100 summaries,
exact recorded-state filtering and an exclusive positive first-event-ID cursor.
It does not infer worker liveness or exportability. Complete escaped history
HTML is capped at 1 MiB; native query strings at 16 KiB.
`GET /explore/run` reads only frozen events through `SQLiteEvidenceReader`,
reusing the 100-event, 1-MiB-per-payload and 8-MiB-total UTF-8 read bounds.
The existing 4-MiB evidence HTML cap includes inline navigation. No dependency,
provider setting, database table or environment variable is added. See
[the complete browser/API/Python guide](guides/SAVED_RUN_BROWSER_GUIDE.md).

Optional `question` filtering accepts 1-200 nonblank Unicode code points,
case-sensitively and literally against the complete first PLANNING question.
The summary remains 300 code points; no new model/index/schema or environment
setting is needed. Search API queries also have a 16-KiB wire bound; full
question decoding and transition checks still scale with saved text/events,
not page size. See [saved-question search](guides/SAVED_QUESTION_SEARCH_GUIDE.md).

`GET /explore/runs/compare` adds a native baseline/candidate chooser, with
1-100 choices per page (default 20), JSON-string exact IDs and structured history
return state. The same 16-KiB query and 1-MiB escaped-page caps apply, plus bounded
frozen events on each side. Either-side errors reject all results. There are no
new settings, dependencies, tables or provider changes; the original JSON
comparison API is unchanged. See the
[workbench contract and measured demo](guides/SAVED_RUN_COMPARISON_WORKBENCH_GUIDE.md).

## Native saved-answer review bounds

The [native review workbench](guides/SAVED_ANSWER_REVIEW_WORKBENCH_GUIDE.md) uses
the configured saved-event/review database, without new settings, dependencies,
tables or provider changes. Query/body limits are 65,536/524,288 bytes;
selection/submission decoded limits are 16,384/262,144 characters. Preview has
four required fields plus at most 50 distinct actual frozen-source checkboxes.
Comments retain the existing nonblank 1-4000-code-point contract. Both the
complete escaped preview and precise persisted page must fit 1 MiB before a
write; generated return queries must fit 16,384 bytes. Exact UUID lookup
preflights one saved row and enforces 262,144 UTF-8 text bytes.

Same-origin form checks require the actual external scheme/host/port, including
explicit ports and IPv6. Trust only configured deployment proxies/root paths;
arbitrary forwarded-host headers cannot redefine the origin. These are local
browser safeguards, not authentication. Existing JSON review APIs, ordinary
notebook pagination and runtime model defaults are unchanged.

## Portable human-review notebook bounds

`GET /runs/{run_id}/notebook` and the exact-ID query-addressed native download
read completed frozen events plus both complete opinion histories in one
read-only SQLite transaction. Fixed maxima: 1,000 reviews, 1,000 annotations,
262,144/16,384 UTF-8 text bytes per review/annotation row, 4,194,304 combined
history text bytes, and 8,388,608 bytes per complete JSON/Markdown artifact.
Both formats must fit before either succeeds; every count/byte overflow is 413.
The existing 100-event, 1-MiB payload and 8-MiB event-text limits also apply.
There are no new environment variables, tables, providers or retention jobs.
The original JSON page APIs and independently paginated browser remain unchanged.
See [the complete notebook contract](guides/PORTABLE_REVIEW_NOTEBOOK_GUIDE.md).

## Collection-browser bounds

The [native collection browser](guides/COLLECTION_BROWSER_GUIDE.md) needs no new
setting, provider or table. Its 1-100-row pages (default 20) project existing
summaries or a complete screening JSON snapshot, capped at 524,288 UTF-8 bytes
even for filtered pages. Complete escaped HTML is capped at 1 MiB.
List/detail queries are at most 16,384 bytes; downstream document/context/search/
worksheet requests carrying validated collection return state allow 65,536 bytes
for percent-encoded Unicode. Ordinary worksheet GET queries remain limited to
16,384 bytes and POST bodies to 65,536. Return state is bounded JSON data, not
an external URL or access token. Existing download bounds are unchanged.

The separate [native collection builder](guides/NATIVE_COLLECTION_BUILDER_GUIDE.md)
reuses the document catalog and atomic collection store without new settings:
1-100 checked exact IDs on one UI page, name 1-120 characters after trimming,
16-KiB query, 256-KiB body, 101 indexed form fields and 1-MiB escaped HTML.
Its same-origin POST saves metadata only; GET and pagination never save or
preserve a multi-page cart. The guide covers Origin/proxy setup and privacy.

The [native editor](guides/NATIVE_COLLECTION_EDITOR_GUIDE.md) uses the same
catalog/name/identity rules. It requires the exact collection and expected
revision, always displays full membership for removals, and paginates only
addition candidates. Up to 100 additions and 100 removals must yield 1-100
members. Native POST permits at most 204 fields and 512 KiB; GET stays at 16 KiB
and complete escaped HTML at 1 MiB. Preview/save both require exact same-origin
headers. No environment setting, server-side draft, dependency or table is added.

## Screening-history bounds

[Human decision history](guides/SCREENING_HISTORY_GUIDE.md) needs no new dependency,
environment variable, provider or model default. Startup adds a separate table,
never copying legacy latest-only decisions into it. `list_history` and its GET
readers require the current collection revision and exact current member ID.
Pages contain 1-100 decisions (default 20), newest revision first with an exclusive
revision cursor. Complete JSON is at most 524,288 UTF-8 bytes; escaped HTML at most
1 MiB. API queries allow 16,384 bytes and native history navigation 65,536 bytes.
Each stored JSON record retains the 16,384-character bound; reasons remain at
most 1000 characters without truncation. No maximum history age/count or purge
setting is added; per-read bounds are not a disk-retention policy.

## Provider Model IDs

The [saved bibliography](guides/SAVED_BIBLIOGRAPHY_GUIDE.md) endpoint has no model
or environment configuration: it reads completed frozen events without provider
calls or current-corpus enrichment. Its fixed source/read limits and exact
262,144-byte JSON and BibTeX ceilings apply regardless of runtime model settings.

| Environment variable | `Settings` field | Default API model ID |
| --- | --- | --- |
| `SCHOLAR_RAG_OPENAI_MODEL` | `openai_model` | `gpt-6-astra` |
| `SCHOLAR_RAG_ANTHROPIC_MODEL` | `anthropic_model` | `claude-sonnet-5-5` |
| `SCHOLAR_RAG_GEMINI_MODEL` | `gemini_model` | `gemini-3.8-flash` |
| `SCHOLAR_RAG_KIMI_MODEL` | `kimi_model` | `kimi-k3.1-pro` |

Settings precedence is explicit `Settings(...)` arguments, process environment,
`.env`, then the built-in defaults. Model IDs are trimmed and must be non-empty
strings; explicit empty, whitespace-only, null, or non-string values fail
validation rather than silently reverting to a default. Leave a variable unset
to use its default. Validation also applies when no provider key is configured.

For example, select the optional Terra model ID without changing the
provider-routing policy or making a request:

```bash
export SCHOLAR_RAG_OPENAI_MODEL=gpt-6.1-terra
```

Or set `Settings(openai_model="gpt-6.1-terra")` in Python. `build_model_router`
passes each field to its corresponding adapter only when that provider has a
key. Direct adapter callers can still pass `model="..."` explicitly; constructors
do not load or override that choice from application environment settings.
Record the previous exact selection before editing and restart after changing
configuration. Follow the
[selection and migration checklist](guides/PROVIDER_MODELS_GUIDE.md#selection-and-migration-checklist)
for account checks, bounded permitted-data evaluation, saved request/provider
identity, and rollback. An ID override alone does not force `/query` away from
a configured Anthropic reasoning provider.

These fields accept custom IDs, not a hard-coded catalog allowlist. The selected
model must support the adapter's endpoint and bounded, single-turn text payload.
No model discovery, account-entitlement check, or inference call happens during
settings validation. See the [provider model guide](guides/PROVIDER_MODELS_GUIDE.md)
for the **2026-10-28 America/Los_Angeles** four-provider catalog recheck,
the separate **2026-10-24 America/Los_Angeles** Kimi K3.1 migration-contract check,
the separately dated **2026-10-01 America/Los_Angeles** Anthropic migration-contract check,
and model-specific limitations. Google's latest flagship is now `gemini-4-argon`;
the selected SPEED-route default remains `gemini-3.8-flash`. OpenAI 6.1 Sol,
6.1 Terra, and Anthropic Haiku 5.5 are catalog alternatives, not live
adapter-compatibility claims. OpenAI documents Chat Completions without tools
for [6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol) and
[6.1 Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra); tool use
requires Responses, which this adapter does not implement. Terra's supported
reasoning efforts are provider capabilities, not new configuration controls.
Selecting Terra adds no vision, computer use, or reasoning-state replay.
All selected defaults, including `kimi-k3.1-pro`, endpoints, routing, and budgets remain unchanged by
this catalog recheck.

Selecting Argon does not implement its long-decode loop or background execution.
Google recommends longer client timeouts for that workload; this repository
keeps 60-second HTTPX network-operation timeouts and a 60-second reasoning-phase
deadline by default, including retries and waits. A slow or unfinished response
can still fail. No new effort, output-budget, or continuation setting is added.

Retrieval-policy comparison does not use provider settings or add environment
variables. It uses two ordinary previews, retaining each one's actual copied
source/hop/phase limits. Its fixed 500-character query, two-variant, 30-second
overall and 256-KiB complete-export bounds are independent of model output
budgets. Reduce selected scope or `SCHOLAR_RAG_MAX_SOURCE_DOCS` and restart
if the full output cannot fit; no per-variant source-limit override is accepted.
See [the comparison guide](guides/RETRIEVAL_COMPARISON_GUIDE.md).

### Anthropic bounded text and rollback

The selected latest public Sonnet, `claude-sonnet-5-5`, uses
`thinking: {"type": "between_tools"}` with explicit `output_config.effort="medium"`.
This stateless adapter sends no tools, so that setting produces text-only output
without up-front thinking or beta headers. Sonnet 5.5 rejects the older
`thinking.type="disabled"` setting.

The output cap remains `max_tokens=1024` per attempt; no token-budget or effort
environment setting is added. Longer answers can still be explicitly truncated
and fail rather than trigger a larger-budget call or automatic continuation.
The catalog's generally recommended Opus 5.5 and demanding-work Fable 5.1 use
always-on thinking, which can exhaust this total budget before final answer text.

To restore the previous Sonnet contract:

```bash
export SCHOLAR_RAG_ANTHROPIC_MODEL=claude-sonnet-5
```

Restart the application after changing settings. That exact ID retains
`thinking: {"type": "disabled"}` without an effort override. All other older,
custom, Opus, or Fable IDs receive neither Sonnet-specific control.
To return to the new default, remove overrides from both the environment and
`.env`, or explicitly set `claude-sonnet-5-5`; explicit `Settings(...)` arguments
still take precedence. Neither these settings nor offline contract tests check
account access, live answer quality, or whether the budget suits your workload.

### Kimi K3.1 default and rollback

The [K2 retirement notice](guides/PROVIDER_MODELS_GUIDE.md#kimi-legacy-overrides-and-retirement-notice)
affects legacy overrides, not this repository's current default. K2 access is
restricted for newly registered users starting **2026-10-31**; existing users
are unaffected by that immediate restriction. All-K2 retirement is planned by
the end of December 2026, not an exact guaranteed date. Earlier discontinuations
of original K2, K2.5, and `moonshot-v1` remain separate and do not provide safe
rollback targets.

The selected default is `kimi-k3.1-pro`; `kimi-k3.1-pro-highspeed` is an optional
override, not an automatic fallback. Both variants inherit K3's documented
parameter constraints. The adapter keeps Moonshot Chat Completions, the
`MOONSHOT_API_KEY` credential, and a body containing only `model` and `messages`.
It omits unsupported `thinking`, fixed sampling fields, and `reasoning_effort`,
leaving effort at the provider's `max` default. No effort knob, multi-turn
reasoning replay, or tool support is added.

To keep the previous K3 model:

```bash
export SCHOLAR_RAG_KIMI_MODEL=kimi-k3
```

Restart after changing settings. Explicit `Settings(kimi_model="...")` arguments
still win over environment, then `.env`, then the built-in default. Remove older
overrides from those sources or explicitly set `kimi-k3.1-pro` to adopt K3.1 Pro.
Direct `KimiAdapter(model="...", api_key="...")` overrides remain independent of
application settings. Custom IDs are preserved, not verified against an account.
The unchanged 60-second HTTPX timeouts and default reasoning deadline can still
expire on slow reasoning responses; the COST route is not a price guarantee.
Without provider keys the application still uses the deterministic offline fake.

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

For `GEMINI_API_KEY`, Google's current guidance permits auth keys and appropriately
restricted standard keys; **unrestricted standard keys are rejected**. Header
transport does not change key type or restrictions. This adapter still sends
`contents`/`parts` to `generateContent`, while Google's latest REST quickstart
uses `model`/`input` at `/v1beta/interactions`. See the source-linked
[Gemini compatibility notes](guides/PROVIDER_MODELS_GUIDE.md#google-gemini);
the quickstart and our offline tests do not establish live endpoint/model access.

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

The family validation applies to application settings only. Direct Python
`ModelRouter(adapters=..., default_provider=...)` callers can still supply
custom adapter names, with the existing routing and fallback behavior.

This fallback policy selects providers before a call. HTTP failures are surfaced
after the existing transient-error retries; they do not trigger cross-provider
or fake failover.

Each live adapter instance admits 60 HTTP attempts per sliding minute by default,
counting the initial request, retries, and failed requests separately. Retries
wait for the larger of exponential backoff and a valid `Retry-After` seconds or
HTTP-date hint, then for rate capacity. Invalid/expired hints retain exponential
backoff; waiting does not consume retry attempts. The three-retry cap and
generation timeout remain unchanged. Server waits are not shortened to fit a
timeout: the runner cancels them and returns an `ERROR` run without another HTTP
attempt. Direct adapter calls need a caller-supplied async timeout to bound
the complete operation, including waits; no new environment setting is added.
Admission state is
in memory, not shared across instances or workers. See
[provider backoff and rate limits](../SAFETY.md#provider-backoff).

Direct Python callers of `llm.rate_limit.with_backoff` must pass a nonnegative
integer `retries` and a finite nonnegative `int`/`float` `initial_delay_seconds`
representable as a finite float. Booleans, strings, negative values, and
non-finite initial delays raise `ValueError` before the operation, retry predicate,
minimum-delay callback, or sleep runs, including with a zero retry budget.
Zero retries means one initial attempt; zero delay is allowed. The default
three retries and 0.25-second exponential schedule, original operation errors,
cancellation, and provider admission/`Retry-After` behavior are unchanged.

The optional `minimum_delay_seconds` callback runs only before a needed retry.
It must return `None` or a finite nonnegative `int`/`float` representable as a
finite float, excluding booleans. Invalid results raise a value-free `ValueError`
even when the exponential delay would otherwise hide them. Each actual retry
wait is also checked for overflow before sleeping or making another attempt.
Unused future waits are not checked: a large finite wait can still lead to
success, or to the original permanent/exhausted error, without a speculative
overflow failure. Valid minimums only extend the current wait, never reset
exponential progression; finite waits are not capped or coerced.
These are helper arguments, not new environment settings.

HTTP-success bodies must be JSON objects with nonblank final answer text and
must not explicitly signal output/context-limit truncation, nonfinal tool or
continuation output, or a block/refusal. Selected Gemini `CONTINUATION` hops
fail before text extraction, with or without `continuationToken`, including
empty/thinking-only hops. Token presence alone is not a nonfinal signal, and
`MAX_TOKENS` retains its distinct truncation diagnostic.
`ProviderResponseError` reports unusable output without raw body or credential
content; it is not retried or replaced with an empty/fake answer. `/query`
preserves its existing HTTP-200 `ERROR` result contract, so check `result.state`.
Rejected text is not grounded or persisted as a generation or `DONE` answer,
and the failed run cannot be exported as a completed answer.
See [response validation](guides/PROVIDER_MODELS_GUIDE.md#unusable-http-success-responses)
for reasoning-only output, rejected finish reasons, unchanged output budgets,
and persistence behavior. Unknown or omitted finish reasons retain the existing
text contract; this is not a general completeness check or automatic continuation.

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
