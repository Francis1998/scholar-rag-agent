# Provider Models and Compatibility

The official [OpenAI GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model),
[Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra),
[Anthropic](https://platform.claude.com/docs/en/models/overview) catalog,
[Gemini 4 Argon](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon) and
[Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) guides,
and [Kimi model list](https://platform.kimi.ai/docs/models.md)
were read directly **2026-10-31 America/Los_Angeles**. This is a current catalog
check, not a new migration or inference test; separately dated model-specific
contracts below retain their original check dates.
OpenAI positions `gpt-6-astra` for highest
intelligence, `gpt-6.1-sol` for balanced near-Astra performance at lower cost,
`gpt-6.1-terra` for efficient visual reasoning, computer use, and writing,
and `gpt-6-luna` for its fastest, most cost-effective high-volume work.
Anthropic generally recommends `claude-opus-5-5`, lists `claude-fable-5-1` for
demanding reasoning and long-horizon agents, and includes `claude-sonnet-5-5`
and `claude-haiku-5-5` in its current lineup.
Google identifies **Gemini 4 Argon (`gemini-4-argon`)** as its latest flagship,
not Gemini 3.8 Flash. Kimi recommends **`kimi-k3.1-pro`** as its latest flagship
and lists `kimi-k3.1-pro-highspeed` as the optional high-speed variant.

This catalog recheck changes no runtime defaults, endpoints, routing, or budgets.
The existing selections remain `gpt-6-astra`, `claude-sonnet-5-5`,
`gemini-3.8-flash` for the SPEED route, and `kimi-k3.1-pro`.
Anthropic's general Opus 5.5 recommendation
does not replace this repository's selected Sonnet 5.5 text-adapter default.
The other providers' catalog alternatives are not new payload-compatibility
tests or default migrations. In particular, Argon's long-decode and latency recommendations
are not implemented by selecting its model ID; see [Gemini limitations](#google-gemini).
Saved-run inspection requires none of these providers. The
[failed-run triage guide](FAILED_RUN_TRIAGE_GUIDE.md) demonstrates why a provider
HTTP 200 can still become a durable `ERROR`, how internal retries differ from
authorizing a new run, and why updating a model ID never repairs saved history.
Before enabling providers, use the
[executable routing preflight](MODEL_ROUTING_PREFLIGHT_GUIDE.md) to inspect
every preferred/default fallback, demonstrate why a fake default alone is not
offline, and retain synthetic fake evidence without live provider requests.

The Kimi K3.1 migration-contract check is **2026-10-24 America/Los_Angeles**:
both K3.1 variants follow the documented K3 constraints, already satisfied by
the existing `model` + `messages` request. The separate Anthropic Sonnet 5.5
migration-contract check remains
**2026-10-01 America/Los_Angeles** (2026-10-01 UTC), earlier default-model
migration guidance **2026-09-17**, and Opus 5.5 migration notes **2026-09-22**.
Sonnet 5.5's no-tools `between_tools` / medium-effort contract is unchanged.
Today's catalog review does not repeat or redate those migrations. Documentation
checks and existing offline HTTPX contract tests are
**not an account-entitlement check or a live inference test**. Availability,
aliases, prices, latency, and output quality can change; evaluate them for your
account and workload.

## Selected Defaults

| Provider | Requested API model ID | Why this default |
| --- | --- | --- |
| OpenAI | `gpt-6-astra` | Highest-intelligence selection in the [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model); its [model page](https://developers.openai.com/api/docs/models/gpt-6-astra) supports text Chat Completions. |
| Anthropic | `claude-sonnet-5-5` | Latest public Sonnet in the [catalog](https://platform.claude.com/docs/en/models/overview), selected for this bounded, no-tools text adapter. Its model-specific thinking/effort controls preserve the 1024-token cap. The catalog generally recommends Opus 5.5 and lists Fable 5.1 for demanding work; neither replaces this Sonnet choice. |
| Google | `gemini-3.8-flash` | Retained [Flash model](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) for the existing SPEED route, not Google's latest overall model. The Argon catalog update does not change this selection. |
| Moonshot | `kimi-k3.1-pro` | Recommended latest flagship in the [K3.1 Pro quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart.md). Both K3.1 variants retain K3's documented parameter constraints; the existing stateless text payload needs no changes. |

The current [OpenAI GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model)
also positions `gpt-6.1-sol` for balancing speed, cost, and intelligence with
near-Astra performance, and `gpt-6-luna` for focused, high-volume workloads.
These are provider
catalog descriptions
([6.1 Sol model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol)),
not measured tradeoffs or additional adapter-contract tests in this repository;
the selected flagship and routing remain unchanged.

The [6.1 Sol model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
separately reviewed **2026-10-13 America/Los_Angeles**, explicitly supports
Chat Completions **without tool calling**; use Responses
for tools. Its supported reasoning efforts are `low`, `medium`, `high`, `xhigh`
and `max`, not `none` or `minimal`. This is documentation for an optional
alternative, not an endpoint, effort, budget or default migration. No live
account or inference verification was performed.

The prior OpenAI `gpt-5.5` and Anthropic `claude-sonnet-4-6` defaults are older
available models, not claimed retired here. `gemini-3.1-pro-preview` remains a
Pro preview rather than the retained Flash choice.

Use `SCHOLAR_RAG_OPENAI_MODEL`, `SCHOLAR_RAG_ANTHROPIC_MODEL`,
`SCHOLAR_RAG_GEMINI_MODEL`, and `SCHOLAR_RAG_KIMI_MODEL` to select alternatives
without editing source. They are nonblank, whitespace-trimmed strings; see
[configuration](../CONFIGURATION.md) for precedence and credential requirements.
`SCHOLAR_RAG_DEFAULT_MODEL` remains a **provider family**, not one of these IDs.

### Optional GPT-6.1 Terra

The [Terra model page](https://developers.openai.com/api/docs/models/gpt-6.1-terra),
rechecked **2026-10-30 America/Los_Angeles**, names the exact ID `gpt-6.1-terra`.
It documents text/image input, text output, and both Chat Completions and
Responses, but **tool calling requires Responses**. Its reasoning efforts are
`none`, `low`, `medium` (default), `high`, `xhigh`, and `max`.
These are provider capabilities, not implemented application features or
measured speed/quality improvements.

Terra is an optional text-model selection here, not a replacement for
`gpt-6-astra`. This adapter still sends only `model` and text `messages` to
Chat Completions. It sends no reasoning-effort override and exposes no effort
setting, so Terra's documented provider default applies rather than a locally
pinned effort. Selecting its ID adds neither image input, computer use,
tool execution, Responses, nor multi-turn reasoning replay. A catalog check
does not establish this account's access or live payload compatibility.

### Kimi legacy overrides and retirement notice

The [Kimi model list](https://platform.kimi.ai/docs/models.md), checked
**2026-10-31 America/Los_Angeles**, distinguishes today's access restriction
from a later planned retirement:

| Timing | Affected users/models |
| --- | --- |
| Starting **2026-10-31** (today's check date) | K2-series models are no longer available to **newly registered users**. Existing users are unaffected by this immediate access restriction. |
| **Planned by the end of December 2026** | Kimi plans to retire **all K2-series models**. This is a planned window, not a guaranteed exact retirement date. |

These notices do not supersede the separately recorded past discontinuations:
the original `kimi-k2` family on **2026-05-25** (the old local default was
`kimi-k2`, not a preview-suffixed ID), and `kimi-k2.5` / `moonshot-v1` on
**2026-08-31**. Do not interpret the new transition window as restored support
for those discontinued IDs or recommend them as fallbacks.

The repository already selects `kimi-k3.1-pro`; no mandatory runtime migration
is needed for that default. Inspect explicit overrides and legacy saved
configuration for K2 IDs, then use the
[K3.1 Pro quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart.md)
and the checklist below to plan a deliberate switch. The quickstart documents
account access requirements; verify them yourself without assuming a model ID
or an API key grants entitlement. `kimi-k3.1-pro-highspeed` remains optional,
not an automatic fallback. Historical saved-run model identities remain
historical evidence, not settings to rewrite.

## Selection and Migration Checklist

1. **Record the baseline.** Keep the exact currently selected provider/model ID
   and configuration source before editing. Explicit `Settings(...)` arguments
   override process environment, then `.env`, then defaults; direct adapter
   constructor arguments are independent. Changing a default does not remove a
   higher-precedence legacy override.
2. **Check access and adapter fit.** Account entitlement, current availability,
   rate limits, and pricing must be checked by the user with their provider.
   Settings validation performs no discovery or inference. Use only a model
   compatible with the existing text endpoint; a provider's vision/tool
   quickstart does not add those capabilities here.
3. **Apply one explicit override.** For an optional Terra text comparison:

   ```bash
   export SCHOLAR_RAG_OPENAI_MODEL=gpt-6.1-terra
   ```

   This command sets no credential and makes no request. Restart the application
   after a deliberate configuration change; do not put this ID in
   `SCHOLAR_RAG_DEFAULT_MODEL`. `/query` still prefers a configured Anthropic
   adapter, so an OpenAI override alone does not force API answers through Terra.
   Check the selected route before any live comparison.
4. **Evaluate a bounded, permitted sample.** First use offline configuration,
   fake, or mocked-HTTP checks for plumbing; they cannot measure model quality.
   Any live evaluation is user-run and separately authorized: start with a
   small fixed set, such as three representative text questions, using only
   data permitted for that provider. Keep the same query, scope, retrieval
   policy, and existing budgets; inspect final captured evidence for drift.
   Compare citation support, completeness, errors, and measured latency/cost,
   not catalog rankings. Do not silently enlarge budgets or retry with another
   provider to turn a failed candidate into an apparent success.
5. **Keep exact comparison evidence.** Retain baseline/candidate run IDs and
   completed frozen exports, including the captured request, source context,
   and generation's `provider` / `model_name`. Check the recorded provider rather
   than inferring it from an environment variable. The model name is the exact
   configured/requested ID, not a provider-resolved immutable snapshot.
   Use the [saved-run comparison](RUN_COMPARISON_GUIDE.md) to inspect differences;
   preserve failed-run state separately instead of treating an `ERROR` as a
   completed answer. Never put credentials or request headers in evidence.
6. **Keep or roll back explicitly.** Restore the previously recorded, still
   available ID in the configuration source that actually selected it and
   restart. If the baseline used the OpenAI default, that ID is `gpt-6-astra`;
   a custom baseline may differ. Removing only an environment override can
   expose an older `.env` value. For Kimi, remove legacy overrides to use
   `kimi-k3.1-pro`, or select a verified available alternative; do not roll back
   to a discontinued K2 ID. No automatic migration or fallback is added.

## HTTP Payload Contracts

These are stateless, single-turn text adapters built on HTTPX. They do not
implement provider tool loops, streaming, multimodal inputs, or multi-turn
reasoning-state replay. The local-first stack recheck for this review confirms
Python 3.11+, FastAPI, Pydantic, HTTPX, SQLite, and a hand-written agent state
machine, not LangGraph. [Runtime wiring](../../src/api/dependencies.py) uses
deterministic hash embeddings, BM25, lexical reranking, and a co-mention graph;
optional spaCy can improve entity extraction but does not make these learned
semantic retrieval or scientific validation. Installing optional dependencies
alone does not enable a cross-encoder or LLM-backed HyDE in the API. Development
checks use ruff, mypy, and pytest.

Each live `generate()` call takes an owned deep snapshot of `LLMRequest` when
its coroutine starts, before its first suspension. Every HTTP attempt and the
returned citation IDs use that same snapshot, so caller edits to prompt,
context, or the citation-ID list during admission, HTTP I/O, or backoff cannot
change an in-flight invocation. The public request remains mutable and can be
reused for later invocations; creating an unawaited coroutine does not snapshot it.

All four adapters admit each HTTP attempt, including retries and failed requests,
against the per-instance sliding-minute rate limit (default 60). Transient
failures retain the same three-retry cap; exponential delays of 0.25, 0.5, and
1 second are minima. A valid `Retry-After` on HTTP `429`, `500`, `502`, `503`, or
`504` extends the wait when longer, followed by rate admission before the next
request. Transport failures and HTTP responses without a usable hint retain
the existing exponential delays.

### Retry-After and timeouts

[RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.html#section-10.2.3) defines
integer delay-seconds and HTTP-date hints. All three standard HTTP-date forms
are accepted as UTC; remaining time is measured when the failed response is
processed, using the local wall clock, not the response's `Date` header.
Malformed/expired hints and numeric overflow fall back to exponential backoff
without exposing header values. Valid finite hints are not capped or rounded
down. [Claude's rate-limit documentation](https://platform.claude.com/docs/en/api/rate-limits)
is one provider example; the header is not guaranteed on every retryable error.
These retry contracts were checked **2026-10-05 America/Los_Angeles**.

Backoff and admission waits count against the runner's reasoning timeout.
A hint longer than the remaining budget causes a cancelled wait and the
existing `/query` HTTP-200 `ERROR` result, not an early retry, provider failover,
or saved completed answer. Direct `adapter.generate()` callers must supply
their own async timeout for a total deadline; the HTTPX network-operation
timeout does not bound these waits. Cancellation sends no further request.
See [rate-limit scope, parsing bounds, and safety](../../SAFETY.md#provider-backoff).

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
truncation and explicit provider blocks/refusals are also rejected, even when
partial answer text is nonblank.
This does not validate factual correctness or interpret every provider stop reason.

### Malformed answer-text blocks

A selected multipart answer is rejected in full if any declared answer-text
block has missing or non-string text. OpenAI/Kimi and Claude explicitly typed
`text` blocks require a string; OpenAI-compatible implicit blocks with a `text`
field and Gemini non-thinking parts with a `text` field also require a string.
`null`, numbers, booleans, arrays, and objects are not skipped or stringified.
Otherwise a valid first block could become a misleading completed answer after
the malformed remainder was silently discarded.

The fixed diagnostic is `malformed answer text`, without provider content.
It is nonretryable and results in a durable `ERROR` run rather than grounding,
recording, or exporting a completed partial answer. Explicit block/refusal,
truncation, and nonfinal-turn diagnostics retain their precedence. Only the
selected candidate is inspected. Non-answer blocks remain ignored, valid
implicit OpenAI text stays supported, and empty string segments remain valid
when the complete answer is nonblank. This is structural validation, not a
guarantee that a well-formed response is factually correct or complete.

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
The tool-call and Anthropic continuation contracts were checked
**2026-10-03 America/Los_Angeles**. The Gemini `CONTINUATION` contract was checked
**2026-10-13 America/Los_Angeles**, separately from model migration contracts.

| Provider | Rejected signal | Official contract |
| --- | --- | --- |
| OpenAI | `choices[0].finish_reason` is `tool_calls` or `function_call`; or the selected message has a nonempty `tool_calls` list or a `function_call` object. | [Chat Completions](https://developers.openai.com/api/reference/resources/chat) defines tool calls and the deprecated function-call form. |
| Kimi | The same checks, inherited from the OpenAI-compatible parser. | [Kimi Chat Completions](https://platform.kimi.ai/docs/api/chat) documents `tool_calls` and submitting tool results. The deprecated `function_call` guard is inherited compatibility handling, not a claim that current Kimi models emit it. |
| Anthropic | `stop_reason` is `tool_use` or `pause_turn`; or a content block has `type: "tool_use"`. | [Claude stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) require tool results or a continuation request, respectively. |
| Gemini | `candidates[0].finishReason` is `CONTINUATION`, `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, or `TOO_MANY_TOOL_CALLS`; or a selected candidate part contains a `functionCall` or `toolCall` object. | [Long-decode continuation](https://ai.google.dev/gemini-api/docs/generate-content/long-decode), [finish reasons](https://ai.google.dev/api/generate-content#FinishReason), [function calls](https://ai.google.dev/api/generate-content#FunctionCall), and [server tool calls](https://ai.google.dev/api/generate-content#ToolCall). |

These responses raise `ProviderResponseError` with the fixed, provider-labelled
`nonfinal tool or continuation output` diagnostic, never the preamble, continuation
token, hidden thinking, tool name/arguments, or raw stop details. A structural call still fails when a
gateway omits the finish reason or reports a normal stop. Gemini's documented
server `toolCall` requires replay in a subsequent turn, not client execution;
neither continuation path is implemented here. Thought filtering cannot hide
a call. Only the first OpenAI/Kimi choice or Gemini candidate is inspected:
there is no alternate-candidate selection or retry to obtain a final answer.

For Gemini long decode, `CONTINUATION` marks an unfinished hop; a client must
replay the original request with the candidate's `continuationToken` until a
terminal finish reason. A hop may contain partial answer text, empty text, or
only thinking. This adapter rejects the selected candidate's `CONTINUATION`
before text extraction, with or without a token, rather than returning that
hop as an answer. **Token presence alone is not a rejection signal**:
`MAX_TOKENS` can also carry a token and retains the distinct `answer was truncated`
diagnostic; streaming checkpoints can carry tokens too. This adapter implements
neither streaming nor automatic continuation.

Normal responses and legacy gateways that omit a finish reason retain their
existing text contract when no explicit truncation, nonfinal, or blocked/refusal
signal is present.
Null call fields and empty OpenAI-compatible `tool_calls` lists are not calls;
other non-text blocks are not generically rejected. Multipart concatenation,
thought exclusion, citations, configured-model provenance, routing, and rate
limits are unchanged. Other or unknown finish reasons are not exhaustively
validated; absence of a response error is not a completeness or correctness
guarantee. This is still a stateless text adapter, not a tool-loop client.

### Explicitly blocked or refused responses

Nonblank partial or refusal text is not a completed research answer when the
provider explicitly marks it as blocked. The following response contracts were
checked **2026-10-08 America/Los_Angeles**, independently of model catalogs or
migrations. These checks run before answer-text extraction.

| Provider | Rejected signal | Official contract |
| --- | --- | --- |
| OpenAI | `choices[0].finish_reason` is `content_filter`; the selected message has a nonempty `refusal` string; or a multipart content part has `type: "refusal"`. | [Chat Completions](https://developers.openai.com/api/reference/resources/chat) defines filtered output and explicit assistant refusals, including typed refusal content. |
| Kimi | The same checks, inherited from the OpenAI-compatible parser. | [Kimi Chat Completions](https://platform.kimi.ai/docs/api/chat) documents the compatible API shape; this is defensive compatibility handling, not a claim that current Kimi models emit every OpenAI refusal/filter signal. |
| Anthropic | `stop_reason` is `refusal`. | [Claude stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) identifies a declined response. This application does not adopt the documentation's retry/fallback recommendation. |
| Gemini | `candidates[0].finishReason` is `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_RECITATION`, `ESCALATION`, or `PUP_LIMITED_DISABLED`. | [Finish reasons](https://ai.google.dev/api/generate-content#FinishReason) explicitly flag filtered candidate content, image safety/recitation, escalation filtering, or policy-disabled generation. |
| Gemini prompt feedback | A nonblank string `promptFeedback.blockReason` other than `BLOCK_REASON_UNSPECIFIED`. | [Prompt feedback](https://ai.google.dev/api/generate-content#PromptFeedback) says a set block reason means the prompt was blocked; documented reasons include `SAFETY`, `OTHER`, `BLOCKLIST`, `PROHIBITED_CONTENT`, and `IMAGE_SAFETY`. |
| Gemini safety ratings | A `safetyRatings` entry on the selected candidate or prompt feedback has boolean `blocked: true`. | [Safety ratings](https://ai.google.dev/api/generate-content#SafetyRating) explicitly identify content blocked by the rating. Probability alone is not a block. |

All of these raise `ProviderResponseError` with the fixed, provider-labelled
`response was blocked or refused` diagnostic, even when no text or candidate is
returned. Refusal strings and typed refusal parts still fail when a compatible
gateway omits the finish reason or reports a normal stop. A typed refusal part
does not need nonblank refusal text to identify a refusal.

Only the first OpenAI/Kimi choice or Gemini candidate is inspected; a later
blocked candidate cannot invalidate an accepted first candidate, and a later
valid candidate cannot replace a blocked first one. Gemini prompt feedback is
response-wide: a nonblank, non-sentinel block reason remains an explicit block
even if its string is unfamiliar or a candidate is present.

Missing/null/empty refusal fields, unspecified prompt block reasons, false
safety flags, and unknown finish reasons alone do not imply a block. Safety
flags must be boolean `true`, not truthy strings or numbers. Ordinary prose
such as "I cannot answer" is not interpreted as a refusal signal. Unblocked
responses retain whitespace, multipart concatenation, thought filtering,
citation mapping, and configured-model provenance.

The existing `/query` contract still uses HTTP 200 with `result.state: "ERROR"`
for a failed run. Inspect `state` and `error`, not just the HTTP status. The
runner journals the failure without grounding the rejected text or recording a
generation or `DONE` event;
the captured input evidence can remain, but the failed run cannot be exported
as a completed answer. Review the provider/model/output-budget configuration
before explicitly starting another run; do not treat empty, explicitly
truncated, nonfinal tool/continuation, or blocked/refused output as evidence.
No partial/blocked/refused output, continuation tokens, raw reasons/details, hidden thinking, headers, or
credentials are included in the diagnostic. No retry, provider/fake failover,
alternate candidate, continuation, or budget increase is attempted.
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
stateless, non-streaming `contents`/`parts` text request. Generation configuration
and sampling overrides remain omitted. Accepted answer parsing concatenates all
answer-text parts and excludes parts marked `thought: true`.

The **2026-10-13 America/Los_Angeles**
[latest-model guide](https://ai.google.dev/gemini-api/docs/latest-model) now covers
`gemini-4-argon`, with `medium` thinking by default. Its REST quickstart sends `model`/`input` to
`POST /v1beta/interactions`, not `generateContent`. This repository has not
migrated to that API/SDK: it still sends `contents`/`parts` and reads
`candidates` from `generateContent`. The Interactions example is not proof of
this endpoint/model combination's compatibility. The separate
[generateContent long-decode guide](https://ai.google.dev/gemini-api/docs/generate-content/long-decode)
documents Argon REST calls and a client-side continuation loop. That loop,
streaming, and Interactions background execution are not implemented here.
The new finish-reason guard rejects unfinished hops; it does not enable long decode.

Google's Argon migration checklist recommends client timeouts of at least
15 minutes for hard prompts; the generateContent long-decode guide recommends a
20-minute HTTP read timeout. This repository instead retains **60-second HTTPX
network-operation timeouts** and a **60-second reasoning-phase deadline by
default**, including retries and waits. Slow Argon overrides can therefore time
out or return an explicit response error rather than a completed answer.
No timeout, output-budget, thinking-level, endpoint, SDK, or selected-default
change is made. Catalog presence and offline HTTPX regressions do not establish
account entitlement, live compatibility, or workload suitability.

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

The [Kimi quickstart](https://platform.kimi.ai/docs/get-your-api-key.md) recommends
`kimi-k3.1-pro` with `POST https://api.moonshot.ai/v1/chat/completions`. This is
now the shared default for Settings, direct construction, and routed Kimi calls.
The endpoint and `MOONSHOT_API_KEY` bearer credential are unchanged.
The quickstart also documents Responses and Messages interfaces. Those are
provider options, not additional adapters implemented here; changing a model ID
does not switch this integration away from Chat Completions.

The [model parameter reference](https://platform.kimi.ai/docs/api/models-overview.md),
checked **2026-10-24 America/Los_Angeles**, explicitly applies K3's constraints to
both `kimi-k3.1-pro` and `kimi-k3.1-pro-highspeed`:

- `thinking` is unsupported. Reasoning is always enabled; `reasoning_effort`
  supports `low`, `high`, and `max`, defaulting to `max`.
- `temperature=1.0`, `top_p=0.95`, `n=1`, and both presence/frequency penalties
  at `0` are fixed. The provider recommends not sending these fields.

The existing body contains only `model` and system/user `messages`, so no
payload correction is needed. It omits all the controls above, leaving effort
at `max`; no optional effort setting or new dependency is introduced.
Accepted responses expose only final answer `content`, never `reasoning_content`.
Multi-turn/tool use requires preserving complete assistant messages including
reasoning state; that remains outside this stateless adapter's contract.
Timeouts, rate admission, and retries are unchanged. Reasoning may still exceed
the runner's default 60-second deadline; the COST route is a static preference,
not a claim that K3.1 Pro is the cheapest model.

To preserve the previous model, set `SCHOLAR_RAG_KIMI_MODEL=kimi-k3` and restart.
For the optional high-speed variant, set
`SCHOLAR_RAG_KIMI_MODEL=kimi-k3.1-pro-highspeed`; it is not an automatic fallback.
Explicit Settings arguments win over environment, then `.env`, then defaults.
Remove old overrides or explicitly select `kimi-k3.1-pro` to adopt the new default.
Direct constructor overrides, older models, and custom IDs remain supported
configuration inputs, not account-entitlement or universal compatibility claims.
No provider key still means offline fake generation.

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

The optional [`near_duplicate_threshold`](NEAR_DUPLICATE_COLLAPSE_GUIDE.md)
uses the existing local lexical collapser before quotas and that minimum check.
It needs no model or provider change; `/retrieve` still makes zero live or fake
generation calls. Completed queries retain the selected models, existing
endpoints, and payload budgets. Collapse is not a model-quality or scientific
equivalence judgment.

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
Kimi regressions cover K3.1 Pro defaults through AppContainer for every task type,
dotenv/environment/explicit precedence, the high-speed variant, legacy `kimi-k3`
rollback and custom IDs. Exact HTTPX body/header assertions keep unsupported
thinking, fixed sampling fields, tools, and effort overrides out of requests.
Reopened SQLite events retain the accepted requested model without reasoning text,
provider-reported aliases, or credentials.
Live sampling is provider-controlled and is not guaranteed deterministic.
These offline regressions and catalog checks do not verify live authentication,
inference compatibility, performance, or answer quality. This catalog recheck
used no paid model calls or live credentials.
