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

## Provider Backoff

Live adapters use an in-process rate limiter before generation and exponential
backoff for transport failures and HTTP `429`, `500`, `502`, `503`, and `504`.
The default is at most three retries after the initial attempt; permanent client
errors are surfaced without those retries.

The limiter waits on a sliding one-minute history and prunes expired timestamps
after waiting. It is per adapter instance, not a distributed quota or strict
count of HTTP requests: retries occur within the already-admitted generation
call. Provider failures do not automatically switch to another provider or the
fake adapter. See the [routing guide](docs/guides/PROVIDER_MODELS_GUIDE.md).
