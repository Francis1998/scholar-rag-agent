# Paper Chat Memory Guide

![Paper chat memory demo](../assets/paper-chat-memory.gif)

`PaperChatMemory` stores multi-turn scholarly chat turns in SQLite, keyed by
`session_id`, with optional `document_ids` / `chunk_ids` provenance. It is a
provider-agnostic library helper, distinct from the agent event log and retrieval
gates. The provenance IDs are caller-supplied references, not document-access
checks or retrieval filters.

**Library-only, not an integrated HTTP chat feature:** the FastAPI routes, agent
runner, and HTTPX LLM adapters do not automatically store or replay these turns.
Library callers explicitly append turns and pass the formatted context to their
own prompting flow. No provider calls, credentials, SDKs, or LangGraph integration
are required by this helper.

## Usage

```python
from storage.paper_chat_memory import PaperChatMemory

memory = PaperChatMemory("paper_chat.db")
memory.append_turn("s1", "user", "What dataset is used?", document_ids=["paper-a"])
memory.append_turn(
    "s1", "assistant", "The authors use MIMIC-III.", document_ids=["paper-a"], chunk_ids=["c12"]
)
prompt_block = memory.format_context("s1", max_chars=2000)
```

`append_turn` returns the inserted integer row ID. Roles, content, session IDs,
and provenance IDs are whitespace-normalized without mutating the caller's lists.
`get_turns` returns a list of frozen `ChatTurn` values with tuple provenance;
without a limit it returns all session turns in chronological order.
`get_turns(..., limit=N)` returns the newest N turns, still in chronological order.
`format_context` keeps the newest complete lines that fit `max_chars`; it returns
an empty string if the session is empty or the newest line does not fit.

`clear_session` returns the number of deleted rows for one session without
affecting others. It is not a secure-erasure guarantee or a cleanup mechanism for
backups, agent events, or other stored data.

## Connection and transaction scope

Initialization and each `append_turn`, `get_turns`, or `clear_session` operation
that accesses SQLite open their own connection and deterministically close it
before returning or propagating an error. The memory object retains only the
database path, not a connection, cursor, pool, or transaction spanning calls.

The SQLite transaction context is nested inside `contextlib.closing`, so its
commit/rollback handling runs **before** the connection closes. Existing explicit
commit points and SQLite schema-creation semantics are unchanged. SQL, JSON
serialization, and commit failures propagate without being converted to successful
results; failed writes roll back any active transaction before closing.

Reads fetch rows before closing, then decode stored provenance JSON. Malformed
JSON still raises its decode error, but the connection is already closed.
`format_context` delegates to `get_turns` and opens no additional connection.
Blank-session reads/clears, reads with a nonpositive limit, and input-validation
failures do not open a connection. A nonblank session with no matching rows still
opens and closes its read or delete connection.

## Current provider catalog note

Official catalogs checked **2026-10-07 America/Los_Angeles** list:

- [OpenAI](https://developers.openai.com/api/docs/models.md): `gpt-6-astra`,
  `gpt-6.1-sol`, and `gpt-6-luna`.
- [Anthropic](https://platform.claude.com/docs/en/models/overview): Sonnet 5.5
  (`claude-sonnet-5-5`), Opus 5.5, and Fable 5.1.
- [Google](https://ai.google.dev/gemini-api/docs/latest-model): `gemini-3.8-flash`
  is generally available.
- [Moonshot](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart): `kimi-k3`.

These names describe the current catalogs, not a model requirement for local
memory or a live-inference/account-entitlement check. See the
[provider model guide](PROVIDER_MODELS_GUIDE.md) for selected defaults and
separately dated payload/migration checks. This lifecycle fix changes no model
selection, endpoint, routing, token budget, or provider contract.
