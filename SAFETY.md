# Safety Controls

These are local execution and evidence-inspection controls, not production
deployment hardening or scientific validation. Use the API on loopback and
review source permissions before ingesting non-synthetic material.

## Timeout Policy

Retrieval defaults to 30 seconds and reasoning defaults to 60 seconds. Both are
configurable and wrap their asynchronous phase calls with `asyncio.wait_for`.
They are cooperative timeouts, not process-level limits that preempt blocking
CPU work. Runtime failures return an `ERROR` run with an `error` field; `/query`
callers must inspect the body even when the HTTP response succeeds.

Both timeouts must be finite: `NaN`, positive/negative infinity, and overflow
such as `1e999` are invalid. `Settings` and environment/`.env` values must be at
least one second and are validated at startup, before storage/provider setup.
Python `SafetyLimits` and effective `RunConfiguration` allow any finite positive
value, including `0.1` seconds; no additional maximum is imposed. Float coercion
is preserved: numeric strings and integers are accepted, `True` becomes `1.0`,
and `False` is rejected as zero. Invalid values are never defaulted or clamped.

`SafetyLimits` validates on construction and copying. Mutations made before a
run or preview are revalidated before planning, retrieval, or generation:
`run` returns a durable `ERROR`, while preview raises `RetrievalPreviewError`
(`planning_failed`, sanitized HTTP 500). Limits already copied for an in-flight
request remain unchanged by later mutations, and finite effective values remain
round-trippable in evidence exports.

## Scope Bounds

`SCHOLAR_RAG_MAX_SOURCE_DOCS` defaults to 50. Despite its name, the current
executor applies it as a maximum number of retrieved **chunk results**, not a
distinct-document quota. Multiple chunks may come from one paper; this is not a
guarantee of source diversity or comprehensive coverage.

Planner tasks request one to three graph hops, depending on intent.
`SCHOLAR_RAG_MAX_HOPS` defaults to 5 and clamps those tasks; API settings allow
no more than 5. Co-mention traversal is bounded retrieval, not proof of a
multi-step scientific argument. Effective bounds and phase timeouts are copied
at run start and retained in evidence exports.

## Document Selection

Optional `document_ids` restricts all query retrieval paths to selected ingested
papers. Scope is copied before awaits, bounded to 100 supplied IDs of 1-128
normalized characters, and retained in the plan and evidence export. HTTP
`null`, empty/blank or invalid scope is rejected with `422`; unknown IDs match
no chunks and never trigger a whole-corpus fallback. Graph edges from excluded
papers cannot bridge into results. BM25 statistics still use the whole corpus.

This is **corpus selection, not authentication or tenant isolation**. It neither
authorizes document access nor protects run/event/export endpoints. Keep the
service local/trusted. Source/hop/capture bounds and human evidence review still
apply; even an empty context may be sent to a configured generation provider.
See the [document scope guide](docs/guides/DOCUMENT_SCOPE_GUIDE.md) for behavior,
legacy component handling, and the offline demonstration.

## Corpus Discovery

`GET /documents` exposes selectable IDs, bounded titles/source labels, and stored
chunk counts, not document bodies or arbitrary metadata. It does not retrieve
evidence, generate answers, or create run events. Titles and source labels may
still contain sensitive data: `no-store`/`nosniff` response headers are not access
control. Keep this read-only endpoint local/trusted like the rest of the API.
Invalid stored IDs fail explicitly rather than returning a truncated selection.
See [document catalog bounds and privacy](docs/guides/DOCUMENT_CATALOG_GUIDE.md).

## Saved Paper Collections

`/collections` manages metadata only. Named selections are local/trusted, not
access-control boundaries. Creation/replacement validates 1-100 normalized
document IDs transactionally; revision preconditions prevent lost edits and
stale deletion. Names and IDs may contain sensitive information even though
the collection endpoints do not return document bodies.

`collection_id` and `document_ids` are mutually exclusive on `/query` and
`/retrieve`. Resolution checks current membership and corpus existence before
awaits, forwarding an immutable scope. Unknown collections, corruption, missing
members, and SQLite failures are explicit errors, never whole-corpus fallbacks.
The existing maximum 50 evidence chunks and other safety bounds still apply.

Metadata edits/deletion do not modify corpus text, graph data, or saved evidence.
They cannot revoke a running request's already resolved selection, and are not
privacy erasure. Membership snapshots do not freeze source contents or add
multi-worker index synchronization. Successful metadata/collection-query
responses and operational errors carry no-store/nosniff headers; default
validation handling is unchanged. See the complete
[collection contracts](docs/guides/PAPER_COLLECTIONS_GUIDE.md).

## Retrieval Preview

Generation-free inspection through `/retrieve` or `AgentRunner.preview` uses the
same copied retrieval, source, hop, and capture bounds. Its reasoning timeout
bounds context preparation only. Preview never invokes a live or fake generator,
grounds claims, or writes agent events; configured LLM-backed HyDE is rejected.
Unlike `/query`, preview reports operational failures as sanitized HTTP 500/504
errors (409 for generative retrieval), not `ERROR` run bodies or empty successes.
Task cancellation propagates without journaling. See the
[retrieval preview contract](docs/guides/RETRIEVAL_PREVIEW_GUIDE.md).

## Cancellation

Python callers can pass a `CancellationToken` to `AgentRunner.run`. It is checked
before planning, retrieval, reasoning, and the final answer transitions.
Token cancellation produces an `ERROR` run; it is not polled inside every
retrieval loop and does not interrupt an in-flight provider call. The HTTP API
does not expose a cancellation endpoint.

Cancelling an already-started `asyncio.Task` running `AgentRunner.run` is different:
once cancellation reaches the runner, it persists one terminal `ERROR` transition
with `agent run was cancelled` and any supplied cancellation message, then
re-raises the original `asyncio.CancelledError`. It does not return a run result,
continue to later phases, or retry generation. Phase timeouts still return a
labelled timeout `ERROR` result.

Cancellation journaling uses synchronous SQLite writes. If that write fails, the
storage exception propagates with the cancellation as its exception context;
durable completion cannot be guaranteed when storage fails. Cancelling before
the coroutine starts produces no run events. Task cancellation does not
force-stop blocking work or guarantee cancellation of an already-sent remote
provider request.

## Hallucination Guard

`CitationGrounder` keeps a claim's mapped chunk ID when the ID exists in the
retrieved set and claim/chunk text share at least one meaningful term. If any
claim fails that check, the answer is prefixed with `[UNGROUNDED]` and includes
a warning. An empty retrieved corpus therefore produces an ungrounded fake
answer rather than evidence.

This grounding check is **non-stopword token overlap**, not semantic entailment
or scientific proof. Evidence exports preserve this flag and the warnings;
resolving a citation to a saved passage does not validate a conclusion.

`grounded: true` and an empty warnings list can still accompany a false or
misleading answer. Read full source passages, original papers, and conflicting
evidence. The offline fake echoes the question rather than generating research
findings. Optional "verification" and screening helpers are advisory and are not
automatically applied by the API; none constitutes a systematic review, novelty
proof, or medical-decision process.

## Evidence Persistence and Privacy

Preview responses contain full source text, query, paths/metadata, and exact
context even though no agent events are persisted. They are not anonymized or
access-controlled. Keep caller-saved previews private, treat content as untrusted,
and protect existing corpus storage and external logs independently. Successful
and operational-error preview responses use no-store/nosniff headers; validation
errors use FastAPI's default handling. Preview digests are not signatures, claim
verification, or guarantees that a later query on a changed corpus will match.

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

Enabling a live model transmits the query and retrieved context to that provider.
Empty all four model-provider keys for the offline API path in the
[Quickstart](QUICKSTART.md); setting only the default provider to `fake` does not
override a configured preferred provider for every task. Treat paper text and
model answers as untrusted input, not instructions to execute commands.

Downloads use hashed filenames, no-store/nosniff headers, and literal fenced
Markdown for untrusted content. Digests detect inconsistent text, not malicious
rewrites by a database owner. Frozen evidence is neither a signed tamper-proof
record nor a promise of identical future model output. See the
[complete evidence export guide](docs/guides/EVIDENCE_EXPORT_GUIDE.md).

Saved-run comparisons expose bounded previews and exact saved identities/digests,
not anonymized data. They reuse the exporter's validation, fail as a whole on an
invalid side, and use no-store/nosniff headers without adding authentication.
Identity overlap and recorded grounding flags do not measure scientific support
or model quality; differing queries/scopes are explicitly not a fair model A/B
test. Comparison does not retrieve, generate, read current corpus data, or write
events. See [comparison limits and privacy](docs/guides/RUN_COMPARISON_GUIDE.md).

## Provider Credentials

Gemini sends `GEMINI_API_KEY` in the provider-supported `x-goog-api-key` header,
not a `?key=` URL parameter. This removes the URL-based credential leak into
HTTPX errors, runner and `/query` error responses, and new durable `ERROR`
events exposed by `/runs/{run_id}/events`. HTTP failures still include their
status and endpoint; retry behavior is unchanged. This is not general-purpose
error redaction, and **request headers still contain the secret**: do not log
them or assume arbitrary exception/debug dumps are safe to share.

Earlier versions may already have saved URL credentials in SQLite events, logs,
backups, or downloaded artifacts. Rotate/revoke affected Gemini keys and restrict
access to those historical artifacts. This change does not rewrite, backfill,
or automatically delete existing records. See the
[Gemini provider guidance](docs/guides/PROVIDER_MODELS_GUIDE.md#google-gemini)
for the dated authentication source and auth-key migration requirements.

## Provider Backoff

Live adapters use an in-process rate limiter before generation and exponential
backoff for transport failures and HTTP `429`, `500`, `502`, `503`, and `504`.
The default is at most three retries after the initial attempt; permanent client
errors are surfaced without those retries.

Each adapter instance admits at most `requests_per_minute` generation calls
(default 60) in a sliding 60-second window measured by a monotonic clock; the
capacity must be positive. A timestamp expires at exactly 60 seconds of age.
Admission is serialized across concurrent callers on the same event loop. Only
the caller holding the admission lock sleeps; it rechecks the clock and capacity
after every wake, including early wakes, before recording an admission. Other
callers wait for that lock. This allows a burst up to the configured capacity,
not evenly spaced calls or a bound on concurrent in-flight generations.

Task cancellation while queued for the lock or sleeping propagates without
consuming a slot or blocking later callers. Once admitted, a generation keeps
its timestamp even if it fails or is cancelled; admission is not refunded.
The lock is released before provider I/O and retries. Retries occur within that
already-admitted call, so the limit is **not a strict count of HTTP requests**,
a token/spend budget, or an account-wide provider quota.

Limiter state is in memory and separate for every adapter instance; it resets
when the adapter is recreated. Use each instance on a single event loop; it is
not thread-safe and does not coordinate separate workers or processes. It is
not a distributed quota. Provider failures do not
automatically switch to another provider or the fake adapter. See the
[routing guide](docs/guides/PROVIDER_MODELS_GUIDE.md).
