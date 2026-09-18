# Safety Controls

## Timeout Policy

Retrieval defaults to 30 seconds and reasoning defaults to 60 seconds. Both are configurable through environment variables and enforced around async execution.

## Scope Bounds

The default source cap is 50 documents per query. Multi-hop graph traversal defaults to depth 3 and is globally bounded by `SCHOLAR_RAG_MAX_HOPS`, never exceeding 5.

## Cancellation

`CancellationToken` is checked at every state transition and inside retrieval loops. Cancelled runs transition to `ERROR` with a structured payload.

## Hallucination Guard

Generated answers must include claims mapped to source chunk IDs. Claims without supporting retrieved chunks are marked `[UNGROUNDED]`, and the response includes a warning.

This grounding check is **non-stopword token overlap**, not semantic entailment
or scientific proof. Evidence exports preserve this flag and the warnings;
resolving a citation to a saved passage does not validate a conclusion.

## Evidence Persistence and Privacy

New queries persist full final-context source passages in the existing event
database before generation. Capture is bounded to 50 passages, 262,144 UTF-8
context bytes, and a 1,048,576-byte serialized snapshot payload. Oversized or
invalid captures and snapshot write failures fail the run before a model call;
the system does not silently truncate evidence to make an export succeed.

The API has no authentication or tenant isolation. Keep it local/trusted.
Both `/runs/{run_id}/events` and the new export route can expose sensitive source
text, queries, metadata, and answers; a run ID is not access control. Corpus
deletion does not delete saved evidence. Protect the SQLite database, backups,
downloaded artifacts, and their retention independently.

Exports allowlist provider/model identity and operational payloads; they do not
dump settings, credentials, headers, environment, raw provider responses, or
hidden model thinking. Unknown event payloads are explicitly omitted. Sensitive
data supplied *as source/query content* is intentionally retained, not redacted:
review every artifact before sharing.

Downloads use hashed filenames, no-store/nosniff headers, and literal fenced
Markdown for untrusted content. Digests detect inconsistent text, not malicious
rewrites by a database owner. Frozen evidence is neither a signed tamper-proof
record nor a promise of identical future model output. See the
[complete evidence export guide](docs/guides/EVIDENCE_EXPORT_GUIDE.md).

## Provider Backoff

LLM calls pass through per-provider rate limiters with exponential backoff for transient `429`, `500`, `502`, `503`, and `504` failures.

Each `AsyncRateLimiter` enforces a sliding one-minute window of at most
`requests_per_minute` calls. When the window is saturated, `acquire` waits until
the oldest slot ages out, then re-anchors the window to the current clock and
drops expired timestamps before admitting the new request. This keeps the
effective admission rate equal to the configured cap rather than throttling
below it because of stale entries left over from a wait.
