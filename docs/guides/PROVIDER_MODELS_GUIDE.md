# Provider Models and Compatibility

Model catalogs rechecked **2026-09-21 America/Los_Angeles** (2026-09-21 UTC);
migration guidance was checked on 2026-09-17 America/Los_Angeles. This is
documentation verification plus offline HTTPX contract
testing, **not an account-entitlement check or a live inference test**. Availability,
aliases, prices, latency, and output quality can change; evaluate them for your
account and workload.

## Selected Defaults

| Provider | Requested API model ID | Why this default |
| --- | --- | --- |
| OpenAI | `gpt-6-astra` | Current flagship in the [model catalog](https://developers.openai.com/api/docs/models.md); its [model page](https://developers.openai.com/api/docs/models/gpt-6-astra) supports text Chat Completions. |
| Anthropic | `claude-sonnet-5` | Current Sonnet, a like-for-like successor to the previous Sonnet default. The [catalog](https://platform.claude.com/docs/en/models/overview) recommends Opus 5 generally and lists Fable 5.1 for the most demanding work; Sonnet 5 is not the absolute newest Claude model. |
| Google | `gemini-3.8-flash` | The [latest-model guide](https://ai.google.dev/gemini-api/docs/latest-model) lists this Flash model as generally available. This deliberately changes the default from a Pro preview to Flash, not to a newer Pro. |
| Moonshot | `kimi-k3` | Current flagship in the [Kimi model list](https://platform.kimi.ai/docs/models), replacing the discontinued K2 default. |

The prior OpenAI `gpt-5.5` and Anthropic `claude-sonnet-4-6` defaults are older
available models, not claimed retired here. `gemini-3.1-pro-preview` remains a Pro
preview rather than the current GA Flash choice. Moonshot explicitly lists the
`kimi-k2` series as discontinued on **2026-05-25**; the old local default was
`kimi-k2`, not a preview-suffixed ID.

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

### Google Gemini

The adapter keeps the `v1beta/models/{model}:generateContent` endpoint and a
`contents`/`parts` text request. The
[Gemini 3.8 Flash migration guidance](https://ai.google.dev/gemini-api/docs/latest-model)
removes sampling overrides; this adapter already omits generation configuration.
The provider's default thinking level is `medium`. Answer parsing concatenates
all answer-text parts and excludes parts marked `thought: true`.

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
