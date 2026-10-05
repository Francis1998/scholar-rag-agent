# Provider Models and Compatibility

The four official model catalogs linked below were checked
**2026-10-04 America/Los_Angeles**. This catalog-only check leaves the selected
defaults unchanged: OpenAI lists `gpt-6-astra`, `gpt-6.1-sol`, and `gpt-6-luna`;
Anthropic lists Sonnet 5.5, Opus 5.5, and Fable 5.1; Gemini 3.8 Flash is GA;
Kimi K3 remains the Kimi default. Anthropic's general Opus 5.5 recommendation
does not replace this repository's selected Sonnet 5.5 text-adapter default.

The separate Anthropic Sonnet 5.5 migration-contract check remains
**2026-10-01 America/Los_Angeles** (2026-10-01 UTC), earlier default-model
migration guidance **2026-09-17**, and Opus 5.5 migration notes **2026-09-22**.
Sonnet 5.5's no-tools `between_tools` / medium-effort contract is unchanged.
These are dated documentation checks and the existing offline HTTPX contract tests,
**not an account-entitlement check or a live inference test**. Availability,
aliases, prices, latency, and output quality can change; evaluate them for your
account and workload.

## Selected Defaults

| Provider | Requested API model ID | Why this default |
| --- | --- | --- |
| OpenAI | `gpt-6-astra` | Current flagship in the [model catalog](https://developers.openai.com/api/docs/models.md); its [model page](https://developers.openai.com/api/docs/models/gpt-6-astra) supports text Chat Completions. |
| Anthropic | `claude-sonnet-5-5` | Latest public Sonnet in the [catalog](https://platform.claude.com/docs/en/models/overview), selected for this bounded, no-tools text adapter. Its model-specific thinking/effort controls preserve the 1024-token cap. The catalog generally recommends Opus 5.5 and lists Fable 5.1 for demanding work; neither replaces this Sonnet choice. |
| Google | `gemini-3.8-flash` | The [latest-model guide](https://ai.google.dev/gemini-api/docs/latest-model) lists this Flash model as generally available. This deliberately changes the default from a Pro preview to Flash, not to a newer Pro. |
| Moonshot | `kimi-k3` | Current flagship in the [Kimi model list](https://platform.kimi.ai/docs/models), replacing the discontinued K2 default. |

The current [OpenAI catalog](https://developers.openai.com/api/docs/models.md)
also positions `gpt-6.1-sol` for balancing intelligence and cost, and
`gpt-6-luna` for cost-sensitive, high-volume workloads. These are provider
catalog descriptions, not measured tradeoffs or additional adapter-contract
tests in this repository; the selected flagship and routing remain unchanged.

The prior OpenAI `gpt-5.5` and Anthropic `claude-sonnet-4-6` defaults are older
available models, not claimed retired here. `gemini-3.1-pro-preview` remains a Pro
preview rather than the current GA Flash choice. Moonshot explicitly lists the
`kimi-k2` series as discontinued on **2026-05-25**; the old local default was
`kimi-k2`, not a preview-suffixed ID. The same Kimi model list also records
`kimi-k2.5` and the `moonshot-v1` series as discontinued on **2026-08-31**;
those are not supported fallback recommendations.

Use `SCHOLAR_RAG_OPENAI_MODEL`, `SCHOLAR_RAG_ANTHROPIC_MODEL`,
`SCHOLAR_RAG_GEMINI_MODEL`, and `SCHOLAR_RAG_KIMI_MODEL` to select alternatives
without editing source. They are nonblank, whitespace-trimmed strings; see
[configuration](../CONFIGURATION.md) for precedence and credential requirements.
`SCHOLAR_RAG_DEFAULT_MODEL` remains a **provider family**, not one of these IDs.

## HTTP Payload Contracts

These are stateless, single-turn text adapters built on HTTPX. They do not
implement provider tool loops, streaming, multimodal inputs, or multi-turn
reasoning-state replay. The local application uses FastAPI, SQLite, Pydantic, and
a custom agent state machine; it is not a LangGraph integration.

All four adapters admit each HTTP attempt, including retries and failed requests,
against the per-instance sliding-minute rate limit (default 60). Transient
failures retain the same three-retry cap and exponential backoff, then wait for
capacity before sending another request. Cancellation while waiting or backing
off sends no further request. See [rate-limit scope and safety](../../SAFETY.md#provider-backoff).

### Unusable HTTP-success responses

All four live adapters require a JSON object and nonblank final answer text.
Missing/malformed answer envelopes, empty or whitespace-only text, and
thinking-only or tool-only output raise `llm.providers.ProviderResponseError`
instead of returning an empty `LLMResponse`. Invalid JSON and non-object roots
raise the same explicit error type. Valid text keeps its original whitespace,
multipart concatenation, citation IDs, and configured-model provenance.

These response errors are not retried and do not trigger another provider or
the fake adapter. Their messages identify the provider and failure category,
not raw response bodies, tool arguments, hidden thinking, credentials, or request headers.
Transport/HTTP retries remain unchanged. Explicit output/context-limit
truncation is also rejected, even when partial answer text is nonblank.
This does not validate factual correctness or interpret every provider stop reason.

### Explicitly truncated answers

The adapters check the finish reason for the selected response before returning
any answer text:

| Provider | Rejected finish reason | Official contract |
| --- | --- | --- |
| OpenAI and Kimi | `choices[0].finish_reason: "length"` | [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat) and [Kimi Chat Completions](https://platform.kimi.ai/docs/api/chat) |
| Anthropic | `stop_reason: "max_tokens"` or `"model_context_window_exceeded"` | [Claude stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) |
| Gemini | `candidates[0].finishReason: "MAX_TOKENS"` | [Gemini finish reasons](https://ai.google.dev/api/generate-content#FinishReason) |

These responses raise `ProviderResponseError` with a fixed, provider-labelled
`answer was truncated` diagnostic. Partial answers, raw stop details, hidden
thinking, and credentials are not included in the error or saved as completed
answers. No retry, automatic continuation, larger-budget call, alternate
candidate, or provider fallback is attempted. A later candidate's stop reason
does not invalidate an otherwise accepted first candidate.

### Nonfinal tool and continuation output

Visible, nonblank text can be a preamble rather than a completed answer. The
following documented signals are rejected before extracting answer text,
because these adapters send no tools and cannot execute tools or resume a turn.
This response-contract check was performed **2026-10-03 America/Los_Angeles**,
separately from the catalog and migration checks above.

| Provider | Rejected signal | Official contract |
| --- | --- | --- |
| OpenAI | `choices[0].finish_reason` is `tool_calls` or `function_call`; or the selected message has a nonempty `tool_calls` list or a `function_call` object. | [Chat Completions](https://developers.openai.com/api/reference/resources/chat) defines tool calls and the deprecated function-call form. |
| Kimi | The same checks, inherited from the OpenAI-compatible parser. | [Kimi Chat Completions](https://platform.kimi.ai/docs/api/chat) documents `tool_calls` and submitting tool results. The deprecated `function_call` guard is inherited compatibility handling, not a claim that current Kimi models emit it. |
| Anthropic | `stop_reason` is `tool_use` or `pause_turn`; or a content block has `type: "tool_use"`. | [Claude stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) require tool results or a continuation request, respectively. |
| Gemini | `candidates[0].finishReason` is `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, or `TOO_MANY_TOOL_CALLS`; or a selected candidate part contains a `functionCall` or `toolCall` object. | [Finish reasons](https://ai.google.dev/api/generate-content#FinishReason), [function calls](https://ai.google.dev/api/generate-content#FunctionCall), and [server tool calls](https://ai.google.dev/api/generate-content#ToolCall). |

These responses raise `ProviderResponseError` with the fixed, provider-labelled
`nonfinal tool or continuation output` diagnostic, never the preamble, tool
name/arguments, or raw stop details. A structural call still fails when a
gateway omits the finish reason or reports a normal stop. Gemini's documented
server `toolCall` requires replay in a subsequent turn, not client execution;
neither continuation path is implemented here. Thought filtering cannot hide
a call. Only the first OpenAI/Kimi choice or Gemini candidate is inspected:
there is no alternate-candidate selection or retry to obtain a final answer.

Normal responses and legacy gateways that omit a finish reason retain their
existing text contract when no explicit truncation or nonfinal signal is present.
Null call fields and empty OpenAI-compatible `tool_calls` lists are not calls;
other non-text blocks are not generically rejected. Multipart concatenation,
thought exclusion, citations, configured-model provenance, routing, and rate
limits are unchanged. Other or unknown finish reasons are not exhaustively
validated; absence of a response error is not a completeness or correctness
guarantee. This is still a stateless text adapter, not a tool-loop client.

The existing `/query` contract still uses HTTP 200 with `result.state: "ERROR"`
for a failed run. Inspect `state` and `error`, not just the HTTP status. The
runner journals the failure without recording a generation or `DONE` event;
the captured input evidence can remain, but the failed run cannot be exported
as a completed answer. Review the provider/model/output-budget configuration
before explicitly starting another run; do not treat empty, explicitly
truncated, or nonfinal tool/continuation output as evidence.
The Anthropic adapter's fixed `max_tokens=1024`
is unchanged and no new environment setting is introduced.
The offline fake and the public response schema for custom adapters are unchanged.

### OpenAI

The adapter keeps `POST https://api.openai.com/v1/chat/completions`, using the
selected `model` plus system and user messages. The
[GPT-6 Astra migration guide](https://developers.openai.com/api/docs/guides/latest-model#migration-quickstart)
forbids `temperature`, `top_p`, and `top_logprobs`, plus `logprobs` for Chat
Completions. No sampling overrides are sent. Astra tool calling requires the
Responses API; keeping Chat Completions is appropriate only for this text-only
adapter, not evidence of tool support.

### Anthropic

The adapter keeps `POST https://api.anthropic.com/v1/messages`, one user message,
`x-api-key`, `anthropic-version: 2023-06-01`, and `Content-Type: application/json`.
The fixed output cap remains **`max_tokens=1024` per attempt**. It sends no tools,
assistant prefill, beta headers, or sampling overrides.

The [Sonnet 5.5 migration guide](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide.md)
enables adaptive thinking when the `thinking` field is omitted; thinking and
answer text share `max_tokens`. It also rejects `thinking: {"type": "disabled"}`
with HTTP 400. Merely updating the default ID while retaining the old payload
would therefore break requests.

The adapter now selects controls by **exact API model ID**, independently of
which ID is the configured default:

| Requested ID | `thinking` | `output_config` |
| --- | --- | --- |
| `claude-sonnet-5-5` (default or explicit override) | `{"type": "between_tools"}` | `{"effort": "medium"}` |
| `claude-sonnet-5` (explicit rollback) | `{"type": "disabled"}` | Omitted |
| All other IDs, including older/custom Sonnet, Opus, and Fable | Omitted | Omitted |

The guide's [up-front thinking section](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide#turn-off-up-front-thinking)
states that `between_tools` produces only text **when no tools are sent**. It
accepts `low`, `medium`, or `high` effort without a beta header, but rejects
`xhigh`/`max` and extra thinking fields such as `budget_tokens` or `display`.
The adapter pins the compatible `medium` level explicitly instead of relying
on the model's default `high`. It does not copy the guide's larger example
budgets, increase spend limits, or introduce a configurable effort surface.
Only blocks whose `type` is `text` contribute to the normalized answer.

The unchanged 1024-token cap is **not enough for every answer**. Explicit
truncation, thinking-only output, and malformed responses still fail as described
above. There is no automatic continuation, larger-budget retry, or provider/fake
failover; transient HTTP/transport retries retain their existing bounded policy.

The [catalog](https://platform.claude.com/docs/en/models/overview) generally
recommends [Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/overview)
and positions Fable 5.1 for demanding work. Both use always-on adaptive thinking;
the [Opus migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)
does not permit disabling it. Those IDs receive neither Sonnet-specific control,
but retain the 1024-token total budget, which thinking can consume before final
answer text. An arbitrary ID override is not a universal compatibility migration.
Evaluate endpoint support, account access, and budget/quality for your workload.

For rollback, set `SCHOLAR_RAG_ANTHROPIC_MODEL=claude-sonnet-5` and restart the
application; `Settings(anthropic_model="claude-sonnet-5")` or the explicit adapter
constructor work as well. To restore Sonnet 5.5, remove old overrides from both
the environment and `.env`, or set `claude-sonnet-5-5` explicitly. Existing
configuration precedence and provider routing do not change.

### Google Gemini

The adapter keeps the `v1beta/models/{model}:generateContent` endpoint and a
`contents`/`parts` text request. The
[Gemini 3.8 Flash migration guidance](https://ai.google.dev/gemini-api/docs/latest-model)
removes sampling overrides; this adapter already omits generation configuration.
The provider's default thinking level is `medium`. Answer parsing concatenates
all answer-text parts and excludes parts marked `thought: true`.

The current latest-model REST example sends `model`/`input` to
`POST /v1beta/interactions`, not `generateContent`. This repository has not
migrated to that API/SDK: it still sends `contents`/`parts` and reads
`candidates` from `generateContent`. The Interactions example is not proof of
this endpoint/model combination's compatibility. The guide's migration checklist
still mentions `generateContent` callers, but the offline tests here only verify
our wire contract, not service availability for the selected model and account.

Credential guidance rechecked **2026-09-26 America/Los_Angeles**: the
[official API-key guide](https://ai.google.dev/gemini-api/docs/api-key) and
[latest-model REST example](https://ai.google.dev/gemini-api/docs/latest-model)
use `x-goog-api-key`; this verifies the header convention, not an unchanged
`generateContent` example. The adapter sends `GEMINI_API_KEY` in that header, never
in the URL. This retains the existing endpoint version, model selection,
payload, parsing, and retry policy while preventing URL-based credential
disclosure through HTTPX errors and newly saved run errors. Headers remain
sensitive and must not be logged.

The current API-key guide rejects **unrestricted standard keys**, while explicitly
restricted standard keys continue to work. New AI Studio keys have defaulted to
service-account-bound auth keys since **2026-05-28**. Use an auth key or a standard
key appropriately restricted for the Gemini API; the guide explains
[migration to auth keys](https://ai.google.dev/gemini-api/docs/api-key#migrate-to-auth-key).
Changing header transport neither migrates a key nor applies its restrictions,
and does not verify account/model entitlement. Configure `GEMINI_API_KEY` with a
key appropriate for your project and review its permissions.
All regression calls use mocked HTTPX transport and dummy credentials, not live
authentication. Historical error records are not rewritten or deleted:
rotate/revoke affected keys and restrict access to saved artifacts as described
in [provider credential safety](../../SAFETY.md#provider-credentials).

### Moonshot Kimi

The adapter uses the [Moonshot API](https://platform.kimi.ai/docs/overview) at
`POST https://api.moonshot.ai/v1/chat/completions`. The
[model parameter reference](https://platform.kimi.ai/docs/api/models-overview)
fixes K3 temperature at `1.0`; other values fail, so the adapter omits sampling
parameters rather than retaining the old `temperature=0.1`.

K3 always reasons. This adapter leaves `reasoning_effort` at the provider default
and reads only answer `content`, never `reasoning_content`. The provider supports
`low`, `high`, and `max` effort, but this change adds no effort-setting surface.
Multi-turn/tool use requires preserving complete assistant messages including
reasoning state; that is outside this stateless adapter's contract.

## Routing, Provenance, and Evidence Limits

The [saved-run bibliography handoff](SAVED_BIBLIOGRAPHY_GUIDE.md) is independent
of current provider/model selection. It uses only completed frozen citation
records and captured metadata, without any live or fake generation, DOI lookup,
or current-corpus enrichment. Exporting references is not a model migration,
metadata-verification step, or new assessment of answer quality.

Routing is unchanged: REASONING prefers Anthropic, SPEED Gemini, COST Kimi, and
DEFAULT the configured family. Missing adapters fall back to that family, then
OpenAI, then fake; HTTP errors do not cause cross-provider failover. API synthesis
always requests REASONING. These preferences are not benchmark claims that the
selected Gemini/Kimi models are always fastest or cheapest.

The optional [`min_evidence_documents`](MINIMUM_EVIDENCE_DOCUMENTS_GUIDE.md)
requirement checks final captured document counts before answer synthesis.
Insufficient built-in queries make no live or fake generation calls; sufficient
queries keep this routing and provider payload contract. Opt-in LLM-backed HyDE
is rejected explicitly, not silently replaced. The count is not a model-quality
or scientific-answerability assessment, and no runtime model ID changes are
required to use it. These dated catalog checks do not establish endpoint or
account entitlement; in particular the Gemini Interactions example is not a
live test of this adapter's retained `generateContent` endpoint.

`LLMResponse.model_name` records the **configured/requested ID**, alongside the
unchanged `raw_provider`; it does not substitute a provider-reported alias or
claim a resolved immutable snapshot. Fake responses leave `model_name=None`.
Credentials and headers are not copied into normalized responses.

Offline regressions cover current defaults, settings validation and precedence,
custom IDs reaching routed payloads/URLs, mocked HTTP request/response contracts,
multipart answer extraction without thought text, and unchanged routing/fakes.
Sonnet regressions check exact model-specific controls, unchanged headers and
budget, legacy rollback, and the absence of those controls for unknown/Opus/Fable
IDs. A real local runner using mocked HTTPX and SQLite confirms that the saved
generation model matches the requested wire ID, not a provider-reported alias.
Live sampling is provider-controlled and is not guaranteed deterministic. No
paid calls, live keys, account access, performance, or generation quality were
tested by this maintenance change.
