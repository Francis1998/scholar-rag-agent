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
| `SCHOLAR_RAG_DEFAULT_MODEL` | `openai` | Exact lowercase provider family for DEFAULT tasks and missing-provider fallbacks: `openai`, `anthropic`, `gemini`, `kimi`, or `fake`; not an API model ID. |
| `SCHOLAR_RAG_OPENAI_MODEL` | `gpt-6-astra` | OpenAI Chat Completions model ID. |
| `SCHOLAR_RAG_ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Anthropic Messages model ID. |
| `SCHOLAR_RAG_GEMINI_MODEL` | `gemini-3.8-flash` | Gemini generateContent model ID. |
| `SCHOLAR_RAG_KIMI_MODEL` | `kimi-k3.1-pro` | Moonshot Chat Completions model ID. |
| `PdfOcrHook.min_chars` | `40` | Constructor threshold: stripped pypdf text shorter than this triggers `OcrBackend` (default `NullOcrBackend`). |

Provider keys are optional: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MOONSHOT_API_KEY`, and `SEMANTIC_SCHOLAR_API_KEY`.

`screened_collection` is a request-only scope for `/query` and `/retrieve`, not
an environment setting. Its strict `collection_id` and `collection_revision`
select current human inclusions in one read snapshot, then pass immutable IDs
to existing retrieval. It is mutually exclusive with ordinary `collection_id`
and `document_ids`; empty or stale selections fail rather than widening scope.
No new tables, providers, dependencies or budgets are introduced. See the
[screened-scope guide](docs/guides/SCREENED_COLLECTION_GUIDE.md).

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
optional collapse and quotas, without increasing any limits. Unmet minima stop answer generation;
previews retain evidence and count diagnostics. See
[Minimum evidence documents](docs/guides/MINIMUM_EVIDENCE_DOCUMENTS_GUIDE.md).

`near_duplicate_threshold` is also **request-only** on both endpoints and runner
methods: a finite strict number greater than zero and at most one, not a bool
or numeric string. HTTP null is invalid; omission or Python `None` disables
collapse. The existing collapser runs after reranking, before quotas/minimums,
without extra retrieval or changes to model defaults, endpoints, or budgets.
Threshold `1` means equal meaningful-term sets, not byte-identical passages.
See [the contract and zero-model demo](docs/guides/NEAR_DUPLICATE_COLLAPSE_GUIDE.md).

Retrieval-policy comparison adds no setting or provider requirement.
`POST /research/compare-retrieval` accepts exactly two named policies over one
shared query/scope. It retains each real preview's effective limits, with fixed
500-character query, 50-source-per-preview, 30-second overall and
256-KiB-per-export bounds. Both formats must fit; reduce scope or the existing
`SCHOLAR_RAG_MAX_SOURCE_DOCS` setting and restart if they do not. No shared
setting is changed per request. See the
[API/Python comparison contract](docs/guides/RETRIEVAL_COMPARISON_GUIDE.md).

The [source-context reader](docs/guides/SOURCE_CONTEXT_GUIDE.md) adds no setting
or provider requirement. Exact document/chunk selectors and before/after counts
are request-only: default two neighbors per side, maximum five. The fixed
2,048-chunk document and 8,192-stored-byte per-chunk metadata bounds apply to
ordering validation even for an anchor-only read. Existing query, retrieval,
pagination and generation defaults are unchanged.

The [native worksheet browser](docs/guides/RESEARCH_WORKSHEET_GUIDE.md#native-browser-workflow)
also adds no settings. It reuses the existing five-question, ten-paper,
three-passage-per-cell and 30-second worksheet bounds. Browser requests have a
65,536-byte body, 16,384-byte URL and six-field limit; escaped HTML must fit
1 MiB. JSON/Markdown keep their existing 256-KiB limits. These are fixed request
and output bounds, not provider budgets, and separate submissions rerun retrieval.

Saved bibliography exports add no setting or provider requirement. They read
completed frozen events only, enforce fixed 50-source and 256-KiB-per-format
bounds, and do not use today's model configuration or corpus to enrich fields.
See [the BibTeX/JSON contract](docs/guides/SAVED_BIBLIOGRAPHY_GUIDE.md).

Collection duplicate inspection also needs no new setting or provider.
The collection revision is required per request; `title_threshold` defaults to
`0.92` and must be finite in `[0, 1]`. Its fixed limits are 100 saved members,
512 characters per title/source/DOI string, 16,384 stored metadata bytes per
document, and 256 KiB for each complete JSON/Markdown body. Both formats must fit;
there is no clipping, paging or unscoped fallback. No model defaults or retrieval
behavior change. See the
[read-only API/Python contract](docs/guides/COLLECTION_DUPLICATES_GUIDE.md).

The [collection browser](docs/guides/COLLECTION_BROWSER_GUIDE.md) adds no setting,
dependency or store. List/member pages use 1-100 rows (default 20), a complete
512-KiB screening JSON snapshot and a 1-MiB escaped HTML cap. Native list/detail
queries allow 16 KiB. Existing readers/search/worksheet accept up to 64 KiB when
carrying validated collection return state; ordinary worksheet URLs retain
16 KiB and form bodies retain 64 KiB. These are finite serialization/navigation
bounds, not model budgets or a promise of frozen output across requests.

The [native collection builder](docs/guides/NATIVE_COLLECTION_BUILDER_GUIDE.md)
adds no setting, dependency or schema. It uses the same catalog page/filter
bounds and collection store: 1-100 exact IDs, a trimmed 1-120-character name,
16-KiB GET query, 256-KiB streamed form and 1-MiB escaped HTML. Exact Origin
and optional same-origin Fetch Metadata are required only for its native POST;
the JSON API stays unchanged. Configure trusted scheme/host/root-path correctly
for mounted/proxy use. There is no multi-page state or authentication.

The [native human screening workbench](docs/guides/HUMAN_SCREENING_WORKBENCH_GUIDE.md)
adds no setting, dependency, table or provider change. It uses the configured
SQLite database, fixed five-entry recent history, 64-KiB query/body limits,
four URL-encoded POST fields and the existing 1-MiB escaped HTML cap. Writes
require one Origin exactly matching the request scheme/host/port; when present,
Fetch Metadata must be `same-origin`. Keep loopback binding or deployment-owned
access/host controls. For mounts/proxies configure the actual root path and
trusted external scheme/authority; mismatches fail closed, and arbitrary
forwarded-host headers do not redefine the accepted origin. No built-in
authentication or DNS-rebinding protection is implied. Original JSON API
Origin behavior is unchanged; this is not a global security policy.

The [human screening progress report](docs/guides/SCREENING_REPORT_GUIDE.md)
adds no settings, tables, dependencies or providers. It consumes one complete
512-KiB screening JSON snapshot for up to 100 members, and caps each requested
JSON/literal Markdown report at 1 MiB of serialized UTF-8. API queries are
limited to 16 KiB with one required revision and one optional format.
All counts, exact exclusion reasons and member IDs derive from that snapshot;
oversized or inconsistent input fails without truncation or extra history reads.

The [screening decision history](docs/guides/SCREENING_HISTORY_GUIDE.md) adds only
an additive SQLite table, initialized at startup without backfill. No setting,
dependency or model default changes. Reads require exact member/collection
selectors and the current collection revision, with 1-100-row pages (default
20), a 512-KiB UTF-8 response limit, 16-KiB API query limit, and 64-KiB native
history query/1-MiB escaped HTML limits. The stored history grows with accepted
submissions; bounded reads do not constitute a storage-retention policy.

The [saved-run browser](docs/guides/SAVED_RUN_BROWSER_GUIDE.md) uses the same
event database without new settings. History has 1-100-row pages and a 1-MiB
escaped HTML cap. Opening frozen evidence uses bounded read-only events
(100 events, 1 MiB per payload, 8 MiB total UTF-8 text) and the existing 4-MiB
HTML limit. The 16-KiB native query-string cap is independent of provider
budgets. These reads do not consult models or the current corpus.

The [native comparison workbench](docs/guides/SAVED_RUN_COMPARISON_WORKBENCH_GUIDE.md)
adds no settings. Its GET chooser uses 1-100 saved summaries (default 20);
baseline/candidate IDs are JSON-string values, and structured return state
preserves history filters. The 16-KiB query and 1-MiB complete escaped HTML caps
are fixed. Existing bounded event limits apply separately to both runs, including
later events; either-side errors reject the whole comparison. The original
JSON comparison API and selected models remain unchanged.

Exact frozen-evidence annotations also add no environment setting or provider
requirement. They use an additive table in the existing database, fixed
1000-character note/quote limits, 1-100-row pages, and a 256-KiB UTF-8 response
cap. IDs and quote selectors come from a completed saved snapshot, not current
model settings or corpus contents. See
[the annotation contract and isolated demo](docs/guides/EVIDENCE_ANNOTATIONS_GUIDE.md).

The [saved-answer review workbench](docs/guides/SAVED_ANSWER_REVIEW_WORKBENCH_GUIDE.md)
adds no setting, dependency, provider default or schema. Native queries allow
64 KiB, streamed forms 512 KiB, generated return queries 16 KiB, full escaped
pages 1 MiB, and precise review reads 256 KiB of UTF-8 text. These are fixed
bounds, not new model budgets. Existing comment limits (4000 code points),
review identities and UUID retry/conflict rules apply. Native writes require
the exact same-origin scheme/host/port; configure trusted proxy/root-path
settings rather than weakening checks. The JSON API contract is unchanged.

The [human-review notebook](docs/guides/HUMAN_REVIEW_NOTEBOOK_GUIDE.md) adds no
settings, dependencies or tables. Each opinion history uses 1-100 rows (default
20), independently; complete escaped HTML is limited to 1 MiB and native
queries to 16 KiB. Existing annotation JSON and bounded frozen-event limits
also apply. The notebook-only review reader shares the existing store; the
original review APIs retain their first-terminal-event behavior. No provider
defaults or model budgets change, and no model is needed for these page reads.

The [portable notebook export](docs/guides/PORTABLE_REVIEW_NOTEBOOK_GUIDE.md)
also adds no settings, tables or providers. One read-only snapshot includes
at most 1,000 reviews and 1,000 annotations, with 256-KiB/16-KiB per-row text
limits respectively and 4 MiB of total UTF-8 history text. Existing bounded
event reads apply. Both complete JSON and Markdown must fit 8 MiB each;
overflow returns 413 without truncation. These caps are not storage retention
policies and do not change the original paginated opinion APIs.

Invalid timeouts, including `NaN`, infinities, and overflow such as `1e999`, fail
settings validation at startup; they are not clamped or replaced with defaults.
Python `SafetyLimits` and effective `RunConfiguration` accept finite positive
subsecond values such as `0.1`; no additional maximum is imposed. See the
[timeout policy](SAFETY.md#timeout-policy) for coercion and runtime error behavior.

`SCHOLAR_RAG_DEFAULT_MODEL` (`Settings.default_model`) is case-sensitive and its
parsed value is not trimmed. Blank, unknown, differently cased,
surrounding-whitespace, and API model ID values fail settings validation before
the application container, storage, or providers initialize; they do not silently
select another provider. Leave it unset to retain `openai`.

Unlike provider families, API model IDs are stripped of surrounding whitespace
and must be non-empty strings.
Unset values use the defaults; explicit blank values fail settings validation,
even without API keys. Model IDs alone never enable live calls.

The selected Anthropic default is the latest public Sonnet, not the catalog's
general Opus recommendation. Exact `claude-sonnet-5-5` requests send
`thinking: {"type": "between_tools"}` and `output_config: {"effort": "medium"}`;
without tools this preserves text-only output within the unchanged
`max_tokens=1024` cap. This cap is not enough for every answer. To roll back,
set `SCHOLAR_RAG_ANTHROPIC_MODEL=claude-sonnet-5` and restart the application:
that exact legacy ID still receives `thinking: {"type": "disabled"}` and no
effort override. All other IDs receive neither control, not a compatibility
guarantee for arbitrary models.

API synthesis requests use REASONING routing, preferring Anthropic when its key
is configured. SPEED prefers Gemini, COST prefers Kimi, and DEFAULT uses
`SCHOLAR_RAG_DEFAULT_MODEL`. A missing preferred provider falls back to the
configured default family, then OpenAI, then the offline fake. This is
configuration fallback, not failover after an HTTP error.
Setting `SCHOLAR_RAG_DEFAULT_MODEL=fake` does not disable preferred live
providers: a keyed Anthropic provider still handles REASONING, and keyed
Gemini/Kimi providers still handle SPEED/COST.

See the [provider model guide](docs/guides/PROVIDER_MODELS_GUIDE.md) for
source-linked defaults and payload compatibility limits. The Sonnet 5.5 migration
contract was checked on **2026-10-01 America/Los_Angeles**.
All four official model catalogs were checked again on **2026-10-25 America/Los_Angeles**;
Google now names `gemini-4-argon` as its latest flagship, while this repository
retains `gemini-3.8-flash` for its SPEED route. OpenAI positions `gpt-6.1-sol` as a
balanced near-Astra option and `gpt-6-luna` for speed/cost-sensitive volume;
Anthropic generally recommends Opus 5.5 and lists Fable 5.1 for demanding
long-horizon reasoning and Haiku 5.5 for latency-sensitive work. These catalog
alternatives do not replace the selected `gpt-6-astra` / `claude-sonnet-5-5` defaults.
The earlier Kimi K3.1 migration-contract check remains dated
**2026-10-24 America/Los_Angeles**; the selected default remains `kimi-k3.1-pro`.
Both K3.1 variants use K3's
constraints: no `thinking` field, provider-default `reasoning_effort="max"`,
and fixed sampling values. The existing `model` + `messages` request omits these
controls, so no endpoint, credential, timeout, retry, or rate-limit change is needed.
Use `SCHOLAR_RAG_KIMI_MODEL=kimi-k3` to keep the previous model, or
`SCHOLAR_RAG_KIMI_MODEL=kimi-k3.1-pro-highspeed` for the optional high-speed variant,
then restart. Explicit Settings arguments still take precedence over environment,
then `.env`, then defaults; remove older overrides to adopt the new default.
All selected defaults and separately dated migration contracts are unchanged
by today's catalog recheck.
Documentation and offline HTTPX contract tests do not establish live inference
compatibility or account entitlement.
The screening progress workflow performs zero model calls; this catalog check
does not enable a provider or change reporting's fixed serialization bounds.

Gemini still uses `generateContent`, not the current Interactions quickstart.
Google rejects unrestricted standard keys; auth keys and appropriately restricted
standard keys are covered in the [Gemini compatibility notes](docs/guides/PROVIDER_MODELS_GUIDE.md#google-gemini).
Setting an API key header does not migrate the endpoint or the key's permissions.
An Argon model override does not add long-decode continuation or adopt Google's
longer timeout recommendations. HTTPX network-operation timeouts remain 60 seconds;
the reasoning-phase deadline remains 60 seconds by default, including retries
and waits. See the same compatibility notes before choosing an alternative.

Live adapters reject invalid JSON, absent/blank final answer text, and explicit
output/context-limit truncation with a nonretryable provider-response error.
Malformed declared answer-text blocks also reject the entire selected multipart
answer, rather than silently returning only its valid segments. Known non-answer
blocks and valid empty text segments retain their existing behavior.
The selected Gemini candidate's `CONTINUATION` reason also fails with the existing
`nonfinal tool or continuation output` diagnostic, even for empty/thinking-only
hops or absent tokens. A continuation token alone is not nonfinal; `MAX_TOKENS`
still reports `answer was truncated`.
Explicit provider content filtering, refusal fields, and Gemini prompt/candidate
blocks likewise fail with a fixed `response was blocked or refused` diagnostic,
without retry, alternate-candidate selection, or provider/fake failover.
`/query` records an `ERROR` run rather than a completed empty or known-partial
answer; rejected text is not grounded, journaled as a generation, or exportable
as a completed answer. Captured input evidence can remain. Existing payload
budgets and routing are unchanged; see
[finish-reason handling](docs/guides/PROVIDER_MODELS_GUIDE.md#explicitly-truncated-answers)
and [nonfinal responses](docs/guides/PROVIDER_MODELS_GUIDE.md#nonfinal-tool-and-continuation-output)
and [blocked/refused responses](docs/guides/PROVIDER_MODELS_GUIDE.md#explicitly-blocked-or-refused-responses).

Live rate admission counts every HTTP attempt, including failed requests and
retries, against the default 60 attempts per adapter instance per sliding minute.
Retries honor valid `Retry-After` seconds/HTTP-date hints as a minimum alongside
exponential backoff, then wait for capacity. Invalid hints retain the existing
delays; the three-retry cap and phase timeouts are unchanged. A hint exceeding
the remaining reasoning budget produces a timeout, not an early retry.
Direct adapter callers need their own overall async deadline.
This is not a shared account-wide quota. See
[provider backoff and rate limits](SAFETY.md#provider-backoff).

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the extended reference and local commands.
