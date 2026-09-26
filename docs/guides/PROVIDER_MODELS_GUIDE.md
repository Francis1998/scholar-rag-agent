# Provider Models and Compatibility

Model catalogs rechecked **2026-09-26 America/Los_Angeles** (2026-09-26 UTC);
default-model migration guidance was checked on 2026-09-17 America/Los_Angeles,
and Opus 5.5 migration notes on 2026-09-22. This is
documentation verification plus offline HTTPX contract
testing, **not an account-entitlement check or a live inference test**. Availability,
aliases, prices, latency, and output quality can change; evaluate them for your
account and workload.

## Selected Defaults

| Provider | Requested API model ID | Why this default |
| --- | --- | --- |
| OpenAI | `gpt-6-astra` | Current flagship in the [model catalog](https://developers.openai.com/api/docs/models.md); its [model page](https://developers.openai.com/api/docs/models/gpt-6-astra) supports text Chat Completions. |
| Anthropic | `claude-sonnet-5` | Current Sonnet, a like-for-like successor to the previous Sonnet default. The [catalog](https://platform.claude.com/docs/en/models/overview) recommends Opus 5.5 (`claude-opus-5-5`) generally and lists Fable 5.1 for the most demanding work; Sonnet 5 is not the absolute newest Claude model. |
| Google | `gemini-3.8-flash` | The [latest-model guide](https://ai.google.dev/gemini-api/docs/latest-model) lists this Flash model as generally available. This deliberately changes the default from a Pro preview to Flash, not to a newer Pro. |
| Moonshot | `kimi-k3` | Current flagship in the [Kimi model list](https://platform.kimi.ai/docs/models), replacing the discontinued K2 default. |

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
not raw response bodies, hidden thinking, credentials, or request headers.
Transport/HTTP retries remain unchanged. This does not validate factual
correctness, interpret every provider stop reason, or reject a nonblank partial
answer merely because it reached an output limit.

The existing `/query` contract still uses HTTP 200 with `result.state: "ERROR"`
for a failed run. Inspect `state` and `error`, not just the HTTP status. The
runner journals the failure without recording a generation or `DONE` event;
the captured input evidence can remain, but the failed run cannot be exported
as a completed answer. Correct the provider/model/output-budget configuration
before explicitly starting another run; do not treat empty output as evidence.
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

The adapter keeps `POST https://api.anthropic.com/v1/messages`, a user message,
and `max_tokens=1024`. The
[Sonnet 5 migration guide](https://platform.claude.com/docs/en/models/sonnet-5/migration-guide)
says non-default `temperature`, `top_p`, or `top_k` can return HTTP 400 and
adaptive thinking is enabled by default. Its output limit covers thinking and
answer text together.

For the exact `claude-sonnet-5` ID, the adapter explicitly sends
`thinking: {"type": "disabled"}` to retain the bounded single-turn answer budget.
Other Anthropic IDs receive **no thinking override**, avoiding a new parameter
on older/custom models; the 1024-token limit still applies. Custom models with
default or mandatory thinking can spend that budget before producing an answer,
so an ID override alone is not a universal migration to arbitrary Claude models.
No sampling overrides are sent for either default or custom models. Only blocks
whose `type` is `text` contribute to the final answer.

The catalog's current general recommendation, [Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/overview),
has always-on adaptive thinking. Its [migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)
does not permit disabling thinking. The adapter sends no Sonnet-specific override
for `claude-opus-5-5`, but still limits total output to 1024 tokens. Evaluate that
budget and endpoint compatibility before switching; thinking-only output now
fails explicitly instead of becoming a successful empty answer. The selected
Sonnet default is intentionally unchanged.

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

Routing is unchanged: REASONING prefers Anthropic, SPEED Gemini, COST Kimi, and
DEFAULT the configured family. Missing adapters fall back to that family, then
OpenAI, then fake; HTTP errors do not cause cross-provider failover. API synthesis
always requests REASONING. These preferences are not benchmark claims that the
selected Gemini/Kimi models are always fastest or cheapest.

`LLMResponse.model_name` records the **configured/requested ID**, alongside the
unchanged `raw_provider`; it does not substitute a provider-reported alias or
claim a resolved immutable snapshot. Fake responses leave `model_name=None`.
Credentials and headers are not copied into normalized responses.

Offline regressions cover current defaults, settings validation and precedence,
custom IDs reaching routed payloads/URLs, mocked HTTP request/response contracts,
multipart answer extraction without thought text, and unchanged routing/fakes.
Live sampling is provider-controlled and is not guaranteed deterministic. No
paid calls, live keys, account access, performance, or generation quality were
tested by this maintenance change.
