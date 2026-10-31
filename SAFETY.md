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

Python `SafetyLimits` requires integer `max_source_docs >= 1` and `max_hops >= 0`.
Booleans, floats (including integral floats and non-finite values), strings, and
`None` are rejected without coercion. These configured counts are never repaired
by clamping. Construction and `dataclasses.replace` validate them; mutations made
before run/preview entry fail before analysis, planning, retrieval, or generation.
As with invalid timeouts, runs record a durable `ERROR` with a field-specific
message that omits the invalid value; previews return sanitized `planning_failed`
errors without events. Already copied in-flight limits remain isolated.

Defaults remain 50 sources and 5 hops. The Python limits have no additional upper
caps: configurations such as 100 sources and 8 hops remain valid, subject to the
separate evidence-capture bounds. With valid configuration, negative *requested*
counts still clamp to 1 source or 0 hops. API `Settings` and environment parsing
retain their existing coercion and bounds; this stricter contract is specific to
Python `SafetyLimits`, not a change to saved evidence schemas.

The runner checks the copied effective limit against both executor retrieval
output and the final captured context, even without an evidence policy. Excess
retrieval stops before `REASONING`; an oversized snapshot is rejected before
its event is appended or answer generation starts. Both produce a durable
`ERROR`, never silent truncation or `DONE`. Preview enforces the same bounds
with sanitized errors and no events. Later changes to shared limits do not
alter an in-flight request's bound. Legacy ordinary answer overrides without
capture callbacks remain supported, but their uncaptured context cannot be
validated or exported as an evidence snapshot.

Opt-in `near_duplicate_threshold` accepts finite strict numbers greater than zero
and at most one. It collapses text-term overlap after reranking, before quotas
and minimum-document assessment, without refill or relaxed bounds. Omission
disables collapse; invalid types, HTTP null, NaN/infinity, and invalid ranges fail
before work. Threshold `1` means equal meaningful-term sets, not identical text.
Lexical Jaccard cannot decide semantic equivalence, paper identity, independence,
or scientific truth; it may drop important similar passages and lower the final
document count enough to block generation. Inspect baseline and collapsed
previews. Survivor paths show transformations, not full dropped-cluster histories.
See [the collapse contract and limitations](docs/guides/NEAR_DUPLICATE_COLLAPSE_GUIDE.md).

Requests may opt into `max_chunks_per_document`, a strict integer from 1 to 50.
The frozen per-request policy caps actual document IDs after reranking of the
existing bounded pool; it does not widen scope, fetch replacements, guarantee
distinct papers, or establish study quality, recall, or source independence.
HTTP null, strings, booleans, floats, and out-of-range values are rejected before
work. Unsupported custom executors fail explicitly rather than ignore a quota.
Previews remain generation/event-write free, and old saved runs without a policy
remain readable. See the complete
[per-paper evidence limits guide](docs/guides/PER_PAPER_EVIDENCE_LIMITS_GUIDE.md).

Opt-in `min_evidence_documents` requires a strict integer from 1 to 50 distinct
document IDs in the final captured context, after reranking, optional collapse, and any quota.
An unmet minimum records `ERROR` with no answer or generation call, but retains
the exact potentially sensitive snapshot and count diagnostic in run events.
Previews with an unmet minimum retain evidence without generation or events;
operational preview failures still return errors. No scope widening, replacement
retrieval, or scientific-support guarantee is implied: a passing count proves
neither relevance nor source independence. See
[minimum evidence contracts and privacy](docs/guides/MINIMUM_EVIDENCE_DOCUMENTS_GUIDE.md).

Planner tasks request one to three graph hops, depending on intent.
`SCHOLAR_RAG_MAX_HOPS` defaults to 5 and clamps those tasks; API settings allow
no more than 5. Co-mention traversal is bounded retrieval, not proof of a
multi-step scientific argument. Effective bounds and phase timeouts are copied
at run start and retained in evidence exports.

Graph-neighbour limits count distinct stored lowercase entity keys, not display
aliases. Both edge directions are filtered by document ownership before grouping
and limiting; excluded edges cannot consume eligible slots or supply display
representatives. The smallest eligible spelling in binary order represents each
key, and neighbours are ordered by key. This deterministic order is not a relevance
score and does not increase hop or result limits.

## Document Selection

Optional `document_ids` restricts all query retrieval paths to selected ingested
papers. Scope is copied before awaits, bounded to 100 supplied IDs of 1-128
normalized characters, and retained in the plan and evidence export. HTTP
`null`, empty/blank or invalid scope is rejected with `422`; unknown IDs match
no chunks and never trigger a whole-corpus fallback. Graph edges from excluded
papers cannot bridge into results. BM25 statistics still use the whole corpus.

At the evidence-capture callback, the runner checks both the plan's selection
and captured source IDs against the original request, even without an evidence
policy. A changed plan scope or out-of-scope snapshot produces a durable `ERROR`
before snapshot persistence or subsequent generation; no snapshot or generation
record is saved for that rejected capture. Preview applies the same selection
checks with sanitized failures and no event writes.

For ordinary queries without an evidence policy, legacy answer overrides without
capture callbacks remain supported: retrieval results and returned citations
are checked, but their uncaptured generation context cannot be inspected or
exported. These are execution-contract checks, not a sandbox against arbitrary
extension code or model calls made before capture.

This is **corpus selection, not authentication or tenant isolation**. It neither
authorizes document access nor protects run/event/export endpoints. Keep the
service local/trusted. Source/hop/capture bounds and human evidence review still
apply; without a requested minimum, even an empty context may be sent to a
configured generation provider.
See the [document scope guide](docs/guides/DOCUMENT_SCOPE_GUIDE.md) for behavior,
legacy component handling, and the offline demonstration.

Explicit `screened_collection` scope requires the current collection revision
and selects only current human `include` decisions. Empty, stale, missing-member
or corrupt-review selections fail before agent work; invalid input returns 422,
including under ASGI mounts. It never changes ordinary collection scope or
automatically includes unsure/stale/unscreened members. Selection is copied
before awaits but does not lock or version corpus text; collection revision does
not detect corpus edits. Saved evidence preserves resolved IDs and actual context,
not authenticated reviewer identity or a new screening-history record.
See [screened selection boundaries](docs/guides/SCREENED_COLLECTION_GUIDE.md).

## Corpus Discovery

`GET /documents` exposes selectable IDs, bounded titles/source labels, and stored
chunk counts, not document bodies or arbitrary metadata. It does not retrieve
evidence, generate answers, or create run events. Titles and source labels may
still contain sensitive data: `no-store`/`nosniff` response headers are not access
control. Keep this read-only endpoint local/trusted like the rest of the API.
Invalid stored IDs fail explicitly rather than returning a truncated selection.
See [document catalog bounds and privacy](docs/guides/DOCUMENT_CATALOG_GUIDE.md).

### Browser inspection of the current corpus

`/explore` and `/explore/document` reuse the bounded read-only catalog/chunk
readers without generation, retrieval/reranking, ingestion, collection changes,
or event writes. Native GET forms/links preserve exact filters and document
scope; only empty source/title fields mean omission. Passage order is chunk-ID
order, not paragraph order. Current pages are not frozen evidence across requests.

Untrusted text stays escaped plain text, sources never become external links,
and metadata is not injected. Restrictive CSP denies scripts, base tags, objects
and framing; only the hash of static CSS is allowed. No-store/nosniff and
no-referrer apply to success and handled errors; errors do not echo inputs or
exception details. Full HTML over 1 MiB fails instead of dropping rows.
NUL-bearing labels/passages fail explicitly rather than being changed by HTML.

There is no authentication or tenant isolation. URLs may expose filters/IDs in
history, bookmarks and server/proxy logs; screenshots or saved pages retain
passages despite no-store. Keep the API local/trusted. Startup still initializes
ordinary stores/indexes, and bounded output is not a scan-time guarantee.
See the [explorer guide](docs/guides/CORPUS_EXPLORER_GUIDE.md) for exact bounds,
failure states, restart behavior and the guarded synthetic browser demo.

## Source-Context Reading

`/documents/context`, `/explore/context`, and `SQLiteSourceContext.read` inspect
one exact document/chunk pair in one read-only snapshot, without retrieval,
models, network, event writes or corpus mutation. Counts are 0-5 on each side.
Every chunk's ordering is validated within a fixed 2,048-chunk document cap;
SQL checks an 8,192-stored-byte metadata cap before JSON parsing. Missing,
invalid or duplicate indices/identities fail explicitly, not by guessed ID
order. Only selected chunks' bounded text/provenance prefixes are fetched.

Missing corpus identities are 404; invalid requests 422; invalid/ambiguous
ordering, oversized metadata/documents and invalid evidence 409; unavailable
storage 503. No partial context is returned. JSON and handled HTML errors use
sanitized diagnostics and no-store/nosniff; HTML retains the explorer CSP and
literal-text rules. Native forms are omitted for control-bearing identities
they cannot preserve; exact encoded navigation remains available.

The window is current corpus, not a historical version or a scientific
judgment. Gaps/overlaps remain, long passages carry explicit truncation flags,
and reducing the window cannot bypass whole-document ordering validation.
Unselected text is not validated; startup still initializes stores and loads
retrieval indexes. Bounds are not a hard scan-time guarantee or an access-control
boundary. IDs in GET URLs and saved artifacts can be sensitive. See
[source-context semantics, privacy and limitations](docs/guides/SOURCE_CONTEXT_GUIDE.md).

## Literal Passage Search

`POST /research/search` searches current stored text without retrieval,
generation, or persistence. It accepts a strict 1-200-character literal query,
at most one nonnull document/collection scope, and a strict limit of 1-50.
Unknown explicit IDs match nothing; invalid/missing collection membership
never falls back to the whole corpus. Cursors bind the query and resolved
scope/revision but are not signed authorization tokens or frozen evidence.

Read-only SQLite snapshots, bounded projections, a 256-KiB UTF-8 response cap,
a 4-MiB stored-passage limit, and a cooperative five-second scan deadline bound
the workflow. Invalid/unsupported projected data and storage failures are
sanitized errors, not empty successes. NUL-bearing passages are rejected
instead of silently using SQLite's NUL-terminated excerpt behavior.
Query text, excerpts, labels, IDs, and cursor digests may be sensitive.
No-store/nosniff headers and scope do not provide authentication or tenant
isolation. Exact wording is not scientific support. See the
[complete literal-search guide](docs/guides/LITERAL_SEARCH_GUIDE.md).

The native `/explore/search` page reuses that exact reader, scope validators,
cursor and bounds. It adds no SQL, ranking, model call, write or index.
Only an omitted phrase opens a form without scanning; present empty phrases,
unknown/duplicate form fields, malformed UTF-8 and invalid/empty selected scopes
fail. Whole-corpus mode rejects a supplied nonblank scope ID rather than
discarding it. Exact browser paper IDs are never whitespace-normalized.

Highlights split the returned excerpt at code-point offsets and escape all
three segments separately; only the first occurrence gets a `<mark>`.
Existing CSP, no-store/nosniff/no-referrer and the 1-MiB escaped-HTML cap apply.
Control-bearing values that native inputs cannot preserve disable the form,
not exact encoded links. Context links reuse the exact document/chunk reader;
partial `search_*` return state fails before source reading rather than
widening a back link. A current context prefix need not contain a late match.

Native GET phrases, scopes and cursors appear in URLs/history/bookmarks/logs.
Unlike invalid inputs or exception internals, **validated** query/scope values
are deliberately retained in escaped error-recovery links; those error pages
may therefore also contain accepted search state. Application diagnostics log
codes, not text. These are rendering and data-integrity controls, not
authentication, erasure or tenant isolation. See the
[browser workflow, limitations and guarded offline demo](docs/guides/BROWSER_PASSAGE_SEARCH_GUIDE.md).

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

## Collection Duplicate Candidates

Revision-required duplicate inspection is bounded to 100 saved members, full
512-character titles/source labels/DOI strings, 16,384 stored metadata bytes per
paper, and 262,144 UTF-8 bytes for **each** complete JSON/Markdown report.
SQL size checks precede decoding/JSON parsing; invalid data fails with 409 and
oversized selected fields or complete output with 413, never clipped matches or
partial partitions. Markdown keeps all untrusted labels in a literal block.

The reader uses one read-only snapshot for collection resolution and selected
metadata, no bodies/chunks, network, generation, retrieval or mutations.
Raw case-sensitive IDs are preserved. Similar titles, DOI aliases and transitive
links are not scientific same-work proof, exhaustive recall, or instructions to
delete/hide/relabel papers. Missing/blank/normalization-empty DOI strings are
absent signals; malformed metadata is an error, not a missing signal. Later
requests may see changed document metadata despite an unchanged collection
revision. Explicit human-selected IDs are required to hand a subset to preview.
All responses/errors use sanitized diagnostics and no-store/nosniff, not access
control. Protect exported IDs, labels and DOI strings. See the
[inspection limits and privacy contract](docs/guides/COLLECTION_DUPLICATES_GUIDE.md).

## Human Paper Screening

Collection screening records human opinions, not authenticated approvals,
scientific inclusion decisions, or PRISMA compliance. Only current-revision
`include` labels enter the explicit included-ID selection; all other states are
excluded from that selection. Collection edits make old labels stale, but
changes to corpus text alone are not detected. Callers must review content
changes and must not substitute the whole collection for an empty selection.

The queue and decision writes validate current membership and revisions
transactionally. Reasons are bounded to 1000 nonblank valid-Unicode characters;
pages to 100 members; queue responses to 524,288 UTF-8 bytes. Malformed records,
conflicts and unavailable storage fail explicitly. Screening never generates or
writes agent events. Latest-only queue/export shapes are unchanged; new successful
submissions also append history in the same transaction. Deleting a collection
does not erase its stored decisions, history or papers. Reasons and IDs can contain
sensitive information, and no-store headers are not authorization. See the
[screening guide](docs/guides/PAPER_SCREENING_GUIDE.md) for exact errors and limits.

The [history readers](docs/guides/SCREENING_HISTORY_GUIDE.md) expose original
reasons/timestamps and an explicit first-retained boundary, never fabricated old
events. Legacy latest-only values remain readable/updatable without backfill.
Removed members fail exact membership checks; re-added members' old opinions
remain stale, not current includes. History reads require the current collection
revision and use one read-only snapshot, 1-100-row pages and a 512-KiB JSON cap.
Native history reuses script-free escaped HTML, CSP/privacy headers and a 1-MiB
cap, with query selectors preserving exact IDs and validated local return state.
Historical reasons now survive latest-decision edits: do not assume an update
erases sensitive text. This is application append-only behavior, not authenticated
reviewer identity, tamper evidence, a retention/purge policy, or scientific proof.

### Native human-screening writes

`/explore/screening` is explicitly write-capable on POST, never GET. It reuses
the existing transactional service and both optimistic revisions. The form
requires a fresh explicit decision/reason; no preselection, auto-screening,
retrieval handoff or automatic retry is performed. Stale submissions and failed
history inserts write nothing. A 303 rereads persisted history; success is not
inferred from a supplied confirmation URL. Existing JSON APIs are unchanged.

Reject missing/null/duplicate/different Origin and non-same-origin Fetch
Metadata before reading POST bodies. Compare scheme/authority exactly, without
Referer fallback or trusting arbitrary forwarded-host values. Strict UTF-8
URL-encoded parsing rejects duplicate/unknown fields, ambiguous inner JSON,
malformed escapes, GET bodies, POST queries and inconsistent lengths. Query
and streaming body caps are 64 KiB; reasons retain the service's 1000-character
bound. Opaque hidden IDs are JSON encoded rather than normalized. Escaped
HTML/CSP, no-store/nosniff and frame denial apply to form/error pages.
Only screening pages, redirects and errors use `Referrer-Policy: same-origin`
so native browser POSTs preserve their Origin instead of Chromium sending
`Origin: null` under `no-referrer`. Strict Origin validation is unchanged:
null/missing/cross-origin values are still rejected. The local page URL and
selection query may be sent as a Referer to this same server, never as a
cross-origin referrer; POST reasons remain out of redirect URLs. Existing
read-only and worksheet pages retain `no-referrer`.

This adds no authentication, CSRF session framework or DNS-rebinding defense.
Direct clients can forge Origin. Use loopback/trusted access, allowed-host
controls and carefully configured proxies; never expose an unauthenticated
instance to an untrusted network. Do not enter credentials or personal data:
append-only reasons survive label changes, and POST is not protection from
resubmission state, logs or screenshots. No scripts/external assets, models,
retrieval or agent events run. See the [complete native browser, API and
Python contract](docs/guides/HUMAN_SCREENING_WORKBENCH_GUIDE.md).

## Read-only collection browser

`/explore/collections` and `/explore/collection` expose existing metadata, not
screening approvals. Membership counts never imply included evidence. Detail
uses one complete, byte-bounded screening export snapshot before filtering its
visible page; stale decisions retain their reason without becoming includes.
Separate requests can see changed decisions/corpus at the same collection
revision. Revision-bound downloads fail on collection edits instead of silently
refreshing or changing scope.

These GET reads perform no model, retrieval, network, mutation or event work.
Normal startup still initializes stores and loads indexes. Existing worksheet
POST retrieval is a separate explicit action. Script-free HTML reuses literal
escaping, hash-only CSS CSP, no-store/nosniff/no-referrer and frame denial.
The 512-KiB full JSON and 1-MiB HTML bounds reject partial success. List/detail
queries are capped at 16 KiB; downstream structured collection-return queries
have a dedicated 64-KiB limit to preserve maximum Unicode filters/IDs. Return
state accepts no arbitrary redirect URL, and unknown/duplicate selection fields
fail explicitly. Raw NUL native worksheet fields retain their existing rejection.
There is no authentication or tenant isolation: use loopback/access controls,
avoid secrets in URLs/reasons, and review artifacts before sharing. See the
[complete browser boundaries and demo](docs/guides/COLLECTION_BROWSER_GUIDE.md).

## Explicit native collection creation

The separate `/explore/collections/new` form writes only after an explicit
same-origin POST. GET/filter/page navigation never writes; no papers are
preselected and no multi-page selection is implied. Strict URL-encoded parsing
rejects unknown/repeated fields, malformed percent/UTF-8, invalid lengths,
GET bodies and POST queries. Indexed JSON-string checkbox values preserve exact
opaque IDs without trimming or browser line-ending normalization. Limits are
16-KiB query, 256-KiB streamed body, 101 fields and 1-MiB complete escaped HTML.

The existing store validates all IDs and name uniqueness atomically, including
concurrent creation. Unknown/deleted IDs, duplicate names and storage failures
never save a partial selection. Error pages/logs do not echo private submitted
values. Names/selections persist as metadata, not frozen text or screening.
No model, retrieval, network, corpus change or agent event is involved.

Origin/Host and optional Fetch Metadata checks, same-origin referrer policy,
CSP/escaping/no-store and root-path-safe redirects follow the other native
write surfaces. They are not authentication, tenant isolation or DNS-rebinding
defense. Inspect saved collections before retrying interrupted writes; there is
no idempotency token. The original JSON API remains unchanged. See the
[complete builder contract and privacy limits](docs/guides/NATIVE_COLLECTION_BUILDER_GUIDE.md).

## Explicit native collection editing

`/explore/collections/edit` uses explicit add/remove deltas, not a visible-page
replacement. All unchecked/off-page members survive. GET and POST preview do
not write; final explicit POST save validates the current revision and **all
old members** in the same reserved transaction as replacement. Missing members
fail even when selected for removal. Stale/repeated saves, duplicate/overlapping
IDs, unknown/repeated fields, invalid encoding, no-op/empty/over-limit results
and conflicting names never partially save or silently repair.

Both POST stages reuse exact Origin/Host/Fetch Metadata checks and bounded
UTF-8 form parsing. Limits are 16-KiB query, 512-KiB streamed body, 204 fields,
1538 decoded characters per POST field and 1-MiB complete escaped HTML. JSON
strings preserve exact control-bearing IDs through browser inputs. Shared
no-store/CSP/escaping/sanitized logging and mount-safe redirects are retained.
Preview is not a signed token or reservation; direct clients can explicitly
save valid deltas without visiting it. Authentication and trusted deployment
boundaries remain the operator's responsibility.

Only collection metadata changes: no documents, chunks, frozen evidence,
events or human opinions/history are deleted or rewritten. Every successful
edit, including rename-only, increments revision and makes earlier screening
stale. Removed-member history remains stored but current-member readers no
longer expose it until re-added. No model/retrieval/network/event work runs.
The existing JSON full-replacement API retains its separate recovery contract.
See the [complete editor guide](docs/guides/NATIVE_COLLECTION_EDITOR_GUIDE.md).

## Retrieval Preview

Generation-free inspection through `/retrieve` or `AgentRunner.preview` uses the
same copied retrieval, source, hop, and capture bounds. Its reasoning timeout
bounds context preparation only. Preview never invokes a live or fake generator,
grounds claims, or writes agent events; configured LLM-backed HyDE is rejected.
Unlike `/query`, preview reports operational failures as sanitized HTTP 500/504
errors (409 for generative retrieval), not `ERROR` run bodies or empty successes.
Task cancellation propagates without journaling. See the
[retrieval preview contract](docs/guides/RETRIEVAL_PREVIEW_GUIDE.md).

The native `/explore/retrieve` workbench computes only on an explicit same-origin
URL-encoded POST; GET opens controls without preview work. It reuses that same
request validation, scope resolution and preparation, with strict bounded wire
parsing and no partial HTML on failure. Empty/ambiguous scopes never widen.
All text is escaped, source labels are never external links, and complete HTML
is capped at 1 MiB. CSP/no-store/framing protections apply to errors too.
Same-origin referrer policy preserves native POST Origin; this is not
authentication, rate limiting, DNS-rebinding protection or isolation of custom
Python components. POST data may remain in resubmission state or infrastructure
logs. Source-context links read current corpus text, not frozen preview evidence.
See the [complete workbench privacy and bounds](docs/guides/NATIVE_RETRIEVAL_WORKBENCH_GUIDE.md).

## Retrieval Policy Comparison

`POST /research/compare-retrieval` runs exactly two real previews under one
cooperative 30-second deadline, never generation or event/corpus writes. Strict
nonblank valid-Unicode queries are bounded to 500 characters; optional shared
scope uses the existing 1-100-ID contract. Only baseline/candidate evidence
policies are accepted. Unsupported options, scoped components and generative
HyDE fail, rather than widening scope or changing algorithms.

One collection read captures membership, identity and revision before awaits;
later edits cannot retarget the candidate. This is **not corpus isolation**:
concurrent contents/index/configuration changes can affect either preview.
Full evidence, actual limits and minimum assessments are retained. Empty overlap
is null; an unmet minimum or `no_passages` is not a scientific judgment.

Complete JSON and Markdown must each fit 262,144 UTF-8 bytes. Any phase failure,
invalid provenance, oversized output or timeout rejects the whole comparison;
external task cancellation propagates without a partial result. Validation
omits request values, operational errors omit raw diagnostics, and responses
use no-store/nosniff. Literal Markdown prevents active source formatting but
does not redact full passages, queries, paths, metadata or identifiers.
Downloads are caller-managed, not new saved runs. Keep the local/trusted API
on loopback; scope/headers are not authentication. See the
[comparison contract](docs/guides/RETRIEVAL_COMPARISON_GUIDE.md).

## Research Worksheets

`POST /research/worksheet` requires explicit document IDs or a saved collection,
never an unscoped default. It inspects at most five questions across ten papers,
returning at most three passages per cell with 800-character excerpt prefixes
and explicit truncation flags. Both download formats must fit 262,144 UTF-8 bytes.
An overall cooperative 30-second deadline also covers export validation.

The service reuses generation-free previews and writes no agent events.
It rejects invalid provenance, unknown/stale selections, oversized output, or any
failed cell rather than returning partial success or widening scope.
`passages_returned` and `no_passages` describe retrieval output, not scientific
support or absence of evidence. Model-backed HyDE remains incompatible.

Membership is resolved once; corpus contents and indexes are not frozen.
Downloads contain sensitive questions, identifiers, labels, and source excerpts;
no-store/nosniff headers and literal Markdown are not access control or redaction.
See [worksheet contracts and privacy](docs/guides/RESEARCH_WORKSHEET_GUIDE.md).

The native `/explore/worksheet` form shares these controls and never calls a
fake/live generator or writes corpus, collections or events. GET only opens
controls; read-only POST permits UTF-8 URL-encoded forms up to 65,536 bytes and
six fields. URLs are capped at 16,384 bytes. Unknown/duplicate fields, malformed
encoding, ambiguous or empty selection, and invalid return state fail explicitly.
Escaped HTML is capped at 1 MiB without partial success. Error pages clear
submitted values rather than echoing private input; static source text cannot
execute markup. CSP, no-store, nosniff and no-referrer are not access controls.

Questions stay out of the form's address bar, but POST bodies can remain in
browser resubmission state or infrastructure logs. Source and return filters
still appear in URLs. Saving/printing/screenshots retain data despite no-store.
Source links read current stored passages, not frozen evidence. Choosing another
output format and submitting again reruns retrieval; no worksheet is persisted.

## Cancellation

Python callers can pass a `CancellationToken` to `AgentRunner.run`. It is checked
before planning, retrieval, reasoning, and the final answer transitions.
Token cancellation produces an `ERROR` run.

For the built-in executor and callback-aware custom executors that invoke
`on_context` before generation, the token is also checked at the shared
evidence-capture boundary after context preparation/reranking. Cancellation
observed there produces a durable `REASONING -> ERROR` with
`agent run was cancelled`, without persisting the prepared snapshot, starting
generation, or recording a generation result or `DONE`. This applies with or
without document selection, per-document quotas, near-duplicate collapse, and
a minimum-document requirement, even when that minimum would not be met.

The token is not polled inside every retrieval loop and does not interrupt an
in-flight provider call. If cancellation is first observed after generation,
the earlier snapshot and completed generation record remain persisted, while
the run returns `ERROR` instead of an answer or `DONE`. Legacy ordinary answer
overrides without capture callbacks remain supported, but cancellation during
their preparation can only be observed after the override returns; their
generation is not preemptively stopped. Custom components are trusted, not
sandboxed: the callback cannot prevent arbitrary model calls made before it.
The HTTP API does not expose a cancellation endpoint.

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
a warning. Without a requested minimum, an empty retrieved corpus therefore
produces an ungrounded fake answer rather than evidence.

Direct Python calls with an empty claim list return `ungrounded=True`, empty
`claims` and `citations`, the `[UNGROUNDED]` prefix, and the warning
`No claims were available for grounding.` This applies even when retrieved
passages overlap the answer or the answer is blank. An existing blank-text claim
instead fails the normal support check and keeps its usual unsupported-claim
warning. The grounder does not parse or synthesize claims; `Executor` still uses
the whole response text as one claim when `parsed_claims` is empty.

Meaningful terms exclude common stopwords and require at least one Unicode
letter or number. Symbols or punctuation alone are not support. Greek letters,
non-Latin text, numbers, and scientific tokens such as `C++`, `p53`, `IL-6`,
and `p<0.05` remain eligible, without splitting or broad punctuation normalization.
This filter is shared by lexical-support helpers; the underlying BM25 and hash
embedding tokenizer is unchanged.

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

The native `/explore/runs/compare` workbench uses bounded read-only frozen-event
reads separately for each selected side, including later events, rather than
the legacy unbounded exporter. Same-ID comparisons read once. Exact JSON-string
form values preserve control-bearing IDs; query-addressed source links use that
side's frozen rank. Complete escaped HTML is capped at 1 MiB and queries at
16 KiB; errors never return partial comparisons or echo raw diagnostics.
Hash-only CSS CSP, no-store/nosniff/no-referrer and no active source links do not
provide authentication or erase browser/log history. No-change is not full trace
equality, quality, a safe-release decision or a fair A/B experiment. See the
[native workflow and limits](docs/guides/SAVED_RUN_COMPARISON_WORKBENCH_GUIDE.md).

`GET /runs/{run_id}/corpus-drift` uses a single read-only transaction to validate
saved evidence and compare at most 50 selected current chunks. It never calls a
live/fake generator, retrieves again, writes events, or mutates corpus/exports.
Current text is capped at 262,144 UTF-8 bytes in aggregate and current records
at 1,048,576; saved-event reads also have explicit count/per-record/total bounds.
Corrupt selected chunks, oversized reads, and SQLite failures are errors, not
partial successes or missing-evidence guesses.

Drift responses contain bounded identities and digests, not source text or
metadata values. These still permit identification/equality checks and are not
anonymous. Matching retrieved chunks do not prove whole-paper equality or
scientific validity; unselected rows and document bodies are not audited.
Data and errors use no-store/nosniff headers, without adding access control.
See [the exact drift contract and limitations](docs/guides/CORPUS_DRIFT_GUIDE.md).

## Native Human Verdicts on Saved Answers

The [saved-answer review workbench](docs/guides/SAVED_ANSWER_REVIEW_WORKBENCH_GUIDE.md)
shows the original answer, both warning sets and full frozen sources before
an explicit accepted/needs_revision/rejected decision. Opening and previewing
never write; only Save appends through the existing review service. Optional
citations must be real frozen chunk IDs, not copied current-corpus IDs.
Existing identity/Unicode/comment validation, append-only UUID conflicts and
original timestamps remain authoritative. There is no default acceptance,
authenticated reviewer, consensus, active learning or scientific validation.

Shared native forms reject ambiguous fields, malformed encoding, GET bodies,
POST queries and cross-origin submissions. Authoring queries are at most 64 KiB,
streamed bodies 512 KiB, return queries 16 KiB, complete escaped pages 1 MiB,
and exact opinion reads 256 KiB of UTF-8 text. Save preflights the complete
persisted reader before writing, using maximum server-field widths. Nothing is
clipped. A UUID-addressed GET rereads that opinion rather than a moving page.
Frozen evidence/opinion reads are separate snapshots, not a signed audit trail.

Keep local/trusted access and host/proxy controls: same-origin headers do not
authenticate direct clients or defend against DNS rebinding. Script-free,
hash-only CSS, no-store/nosniff and frame denial remain in place. Authoring
surfaces use same-origin referrers so native POST Origin works; persisted
readers retain no-referrer. Comments stay out of redirect URLs, not browser
resubmission state or infrastructure logs. Screenshots, history and downloads
can retain sensitive content. No corpus, retrieval, provider, event, annotation
or screening work occurs during authoring; startup remains unchanged.

## Human Notes on Frozen Quotes

Saved-run annotations validate exact frozen chunk/document IDs, whole-source
SHA-256, strict Unicode code-point offsets, and the literal source slice.
They do not infer an occurrence from similar text or check scientific support.
UUIDs provide retry identity, not reviewer authentication; every new opinion
has a separate immutable record. Whole-answer reviews and collection screening
remain separate features.

Notes and quotes are limited to 1000 code points, pages to 100 annotations plus
one validated lookahead, and compact JSON to 262,144 UTF-8 bytes. SQL preflights
bound saved events and selected annotation text before hydration, in one
transaction. Unknown, invalid, oversized, stale-digest, or unavailable storage
outcomes are explicit sanitized errors, not partial success. Creation/replay
is atomic and neither operation writes agent events or invokes generation,
retrieval, or current-corpus reads.

Notes and quoted source text can contain private or untrusted content. Display
them literally, never execute instructions or render raw HTML. No-store/nosniff
headers are not authentication, encryption, or redaction. No edit/delete API,
retention job, author identity, or tenant boundary is added. Corpus deletion
does not erase annotations, saved evidence, or backups. Digests detect an
inconsistent anchor, not a database owner's coherent rewrite of all records.
See [exact selectors, bounds, and errors](docs/guides/EVIDENCE_ANNOTATIONS_GUIDE.md).

The [native human-review notebook](docs/guides/HUMAN_REVIEW_NOTEBOOK_GUIDE.md)
reads both histories without adding browser writes, reviewer identities,
consensus or scientific validation. Comments, notes and quotes render as exact
escaped text under the existing script-free CSP. Generated links stay native;
untrusted source URLs never activate. A selected annotation/source pair shares
one validated frozen read snapshot, while the combined notebook explicitly does
not. Bounded events, validated lookahead, a 16-KiB query cap and a 1-MiB escaped
HTML cap fail explicitly, with sanitized logged errors and no partial page.
No generation, retrieval, network, current-corpus reads or event/opinion writes
occur on these GET routes. Normal startup still initializes existing stores.
IDs/return selections remain visible in URLs/history/logs; no-store and
anti-framing headers do not add authentication, retention or erasure.

### Portable complete notebooks

[Notebook JSON/Markdown exports](docs/guides/PORTABLE_REVIEW_NOTEBOOK_GUIDE.md)
read saved events, all retained answer reviews and all retained quote annotations
in one existing-database read-only snapshot. They preserve human decisions
separately from model grounding; no consensus, latest verdict or reviewer identity
is inferred. Source IDs/digests and exact code-point slices are validated using
the existing opinion validators, never a live corpus or a guessed occurrence.

Fixed limits bound each history to 1,000 records, per-row review/annotation text
to 256 KiB/16 KiB, and combined history text to 4 MiB of exact UTF-8. Existing
event read limits apply, including later events. Both complete JSON and literal
Markdown must fit 8 MiB each before any success; every count/byte overflow
returns sanitized 413 without discarding records. Stored-data corruption and
missing tables remain explicit errors. Private headers cover successful and
handled error responses, including 405; filenames hash rather than embed IDs.

Markdown uses the existing variable-length literal fences for every untrusted
value, never active source URLs or instructions. JSON is the exact machine
artifact; a Markdown viewer can still visually normalize controls/whitespace.
Comments, full captured text, queries and metadata can be sensitive. Saving a
download makes a lasting copy; corpus deletion cannot erase it. No-store,
nosniff and no-referrer are not authentication, encryption or redaction. Caps
do not bound history retention, authenticate database authors or promise hard
scan-time deadlines. No new provider, mutation, event write or configuration
knob is introduced, and paginated opinion API contracts remain unchanged.

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

An HTTP-success response is not automatically a successful answer. Live adapters
raise `ProviderResponseError` for invalid/non-object JSON or absent/blank final
answer text after excluding non-answer blocks. This includes thinking-only and
tool-only responses. Explicit output/context-limit truncation also fails, even
with nonblank partial text: OpenAI/Kimi `length`, Anthropic `max_tokens` or
`model_context_window_exceeded`, and Gemini `MAX_TOKENS`.
Explicit nonfinal tool/continuation output also fails. In particular, the
selected Gemini candidate's `finishReason: "CONTINUATION"` is an unfinished
long-decode hop, not a completed answer, even if its text is nonblank.
Empty/thinking-only hops and hops without a `continuationToken` receive the
same fixed `nonfinal tool or continuation output` diagnostic before extraction.
A token alone is not a nonfinal signal: `MAX_TOKENS` can carry one and still
reports `answer was truncated`. See the
[continuation contract checked on 2026-10-13 America/Los_Angeles](docs/guides/PROVIDER_MODELS_GUIDE.md#nonfinal-tool-and-continuation-output).
Explicit content-filter/refusal signals also fail before text extraction:
OpenAI/Kimi `content_filter`, nonempty message refusal strings or typed refusal
parts; Anthropic `refusal`; and Gemini's documented blocked finish reasons,
prompt block reasons, or boolean `blocked: true` safety ratings.
These produce the fixed `response was blocked or refused` diagnostic, including
responses without answer text. Only the selected first choice/candidate is used;
prompt feedback applies to the whole response. See the
[dated blocking contract](docs/guides/PROVIDER_MODELS_GUIDE.md#explicitly-blocked-or-refused-responses).
The diagnostic contains a provider name and fixed failure
category, not raw provider output. These errors are not retried or failed over;
the runner records `ERROR` rather than persisting an empty or explicitly
truncated/nonfinal/blocked completed answer. Rejected text is never grounded or promoted
to a generation record or completed export, and there is no alternate candidate,
automatic continuation, or budget increase.
Existing captured input evidence remains subject to the privacy rules above.
Unknown or missing finish reasons retain the existing text contract when no
explicit block/refusal signal is present; ordinary prose is not classified as
a refusal. Raw refusal text, filter details, hidden thinking, headers, and
credentials or continuation tokens are not included in the fixed diagnostic. This check
does not establish factual correctness or completeness of other nonblank answers.
See [provider response validation](docs/guides/PROVIDER_MODELS_GUIDE.md#unusable-http-success-responses).

Live adapters use an in-process rate limiter before every HTTP attempt and
exponential backoff for transport failures and HTTP `429`, `500`, `502`, `503`,
and `504`. The default remains at most three retries after the initial attempt,
with minimum backoff delays of 0.25, 0.5, and 1 second; permanent client errors are
surfaced without those retries.

On retryable HTTP responses, a valid `Retry-After` can extend each wait to the
larger of the exponential delay and the server hint. Following
[RFC 9110 section 10.2.3](https://www.rfc-editor.org/rfc/rfc9110.html#section-10.2.3),
the header accepts nonnegative ASCII integer seconds or an HTTP-date.
The three [HTTP-date forms](https://www.rfc-editor.org/rfc/rfc9110.html#section-5.6.7)
(IMF-fixdate, obsolete RFC 850, and asctime) are interpreted as UTC, including the
RFC 850 rolling 50-year rule. Dates use the remaining time from the local UTC
wall clock when each failed response is processed, not request-start time or
the response's `Date` header. Keep the host's wall clock synchronized.
The resulting duration is awaited before re-entering monotonic rate admission.

Missing, expired, malformed, duplicate, negative, fractional, nonfinite, and
overflowing hints retain the existing exponential backoff. Numeric hints must
fit a finite Python float; significant digits are bounded before integer
conversion, and representable delays are rounded upward if necessary, never
capped or rounded down to send early. Header contents are not added to error
messages or logs. A hint applies only to that failed attempt: it does not change
later exponential delays, impose an account-wide cooldown, change retryable
statuses, or create a retry on a successful/permanent failure. Transport errors
have no HTTP hint and retain their existing backoff.

Direct `with_backoff` callbacks have a stricter contract than raw HTTP hints:
`minimum_delay_seconds` must return `None` or a finite nonnegative `int`/`float`
representable as a finite float, excluding booleans. Invalid callback results
and overflowed exponential waits raise a value-free `ValueError` before the
affected sleep or another operation attempt; they are not silently ignored or
capped. Validation applies only when a retry wait is needed. A hypothetical
future overflow does not reject a successful retry or replace the original
permanent/exhausted error. Finite waits remain cancellable and have no new cap.

This implements a documented provider contract, not a promise that every error
carries the header. For example, [Claude rate limits](https://platform.claude.com/docs/en/api/rate-limits)
document `retry-after` on rate-limit responses, but not on monthly spend-cap
errors. Both the RFC and provider retry contract were checked
**2026-10-05 America/Los_Angeles**, independently of model migrations.

The `requests_per_minute` argument to `AsyncRateLimiter` and the shared
`HTTPProviderAdapter` constructor must be a positive Python `int`. Invalid
capacities raise `ValueError` naming `requests_per_minute` during construction.
Booleans, strings, all floats (including integral floats, NaN, and infinities),
`None`, other non-integer types, zero, and negative integers are rejected without
coercion. Positive Python integers have no fixed upper bound.

Each adapter instance admits at most `requests_per_minute` HTTP attempts
(default 60) in a sliding 60-second window measured by a monotonic clock; the
initial attempt and every retry each consume one slot, including failed
requests. A timestamp expires at exactly 60 seconds of age. Admission is
serialized across concurrent callers on the same
event loop. Only the caller holding the admission lock sleeps; it rechecks the
clock and capacity after every wake, including early wakes, before recording an
admission. Other callers wait for that lock. This allows a burst up to the
configured capacity, not evenly spaced requests or a bound on concurrent
in-flight requests.

Task cancellation while queued for the lock or sleeping propagates without
consuming a slot, sending a request, or blocking later callers. Once admitted,
an attempt keeps its timestamp even if it fails or is cancelled; admission is
not refunded. The lock is released before provider I/O. After a transient
failure, backoff runs before the next attempt waits for rate capacity; waiting
does not use up retries. Cancellation during backoff sends no further request.
Rate waits and backoff remain subject to the existing generation phase timeout.
If a server wait outlasts the remaining reasoning budget, the runner cancels
the wait instead of shortening it or sending another request. `/query` retains
HTTP 200 with `result.state: "ERROR"` and a labelled reasoning timeout; there is
no generation record, completed answer, or `DONE` event. Captured input evidence
can remain, and completed-answer export stays unavailable for that run.
Direct `adapter.generate()` calls have no overall generation deadline:
HTTPX's 60-second network-operation timeouts do not bound backoff or admission
waits. Direct callers needing a total deadline must wrap the call in an async
timeout. Cancellation always stops the wait rather than forcing an early retry.
The default reasoning-phase deadline is also 60 seconds. These limits have not
been raised to Google's longer Argon latency/long-decode recommendations;
selecting `gemini-4-argon` does not add continuation, streaming, or background
execution. Slow or unfinished Argon responses can fail within the current
runtime. See the [dated Gemini limitations](docs/guides/PROVIDER_MODELS_GUIDE.md#google-gemini).
This per-instance attempt limit is not a token/spend budget or an account-wide
provider quota.

Limiter state is in memory and separate for every adapter instance; it resets
when the adapter is recreated. Use each instance on a single event loop; it is
not thread-safe and does not coordinate separate workers or processes. It is
not a distributed quota. Provider failures do not
automatically switch to another provider or the fake adapter. See the
[routing guide](docs/guides/PROVIDER_MODELS_GUIDE.md).
