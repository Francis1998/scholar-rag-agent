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

`clear_session` returns the number of deleted rows for one session without
affecting others. It is not a secure-erasure guarantee or a cleanup mechanism for
backups, agent events, or other stored data.

## Context character budget

`format_context(session_id, max_chars=4000)` requires a positive Python `int`,
excluding `bool`, with no upper cap. Zero, negative integers, all floats
(including integral floats, NaN, and infinities), strings, `None`, and other
non-integer values raise `ValueError("max_chars must be a positive integer")`.
Validation occurs before `format_context` calls `get_turns` or accesses SQLite,
even for a missing or blank session.

The budget counts Python characters, not bytes or model tokens, including role
prefixes and newline separators. Formatting keeps the newest contiguous suffix
of complete turns in chronological order; it stops at the first turn that does
not fit rather than skipping it to include older, shorter turns. It returns an
empty string if the session is empty or the newest turn exceeds the budget.
The default remains 4000 characters; `get_turns(..., limit=N)` is unchanged.

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

## Provider compatibility

This helper is provider-agnostic and performs no model calls. See the shared
[provider model guide](PROVIDER_MODELS_GUIDE.md) for selected defaults and
separately dated catalog and payload-compatibility notes. Context validation
changes no model selection, endpoint, routing, or provider token budget.
