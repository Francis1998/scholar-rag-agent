# Inspect human screening decision history

See how your own `include`, `exclude`, and `unsure` opinions changed, including
their exact reasons, decision revisions, reviewed collection revisions, and
original recorded timestamps. New successful submissions append a separate
history record **in the same transaction** as the existing latest decision.
There is no new editing UI, automatic label, model call, or evidence selection.

![Executed synthetic screening-history output illustration](../assets/screening-history.gif)

This GIF illustrates **executed synthetic output**, not browser screenshots,
a recording, a scientific benchmark, or a completed literature review. Its
four panels come from the reproducible offline demo below, using the existing
Pillow renderer. The latest-only legacy row is an explicitly seeded synthetic
fixture, not a recovered historical event.

## End-user workflow

1. Start the loopback service using the [Quickstart](../../QUICKSTART.md).
   Ingest permitted papers, save a [collection](PAPER_COLLECTIONS_GUIDE.md), and
   use the existing [human screening API/Python interface](PAPER_SCREENING_GUIDE.md)
   to record decisions. Existing submission, queue and export shapes are unchanged.
2. Open `/explore`, choose **Browse saved collections**, then inspect a
   collection. Every member has **Inspect human decision history**, including
   members with no decisions. The link binds the exact member and displayed
   collection revision, not a document-ID path suffix.
3. Read the **retained history boundary**, the **latest decision in this read**,
   and the newest-first decision cards. A historical `include` is not evidence
   that the member is currently included. The page's `current_status` reflects
   the latest decision against the current collection revision.
4. Follow **Next page** for older decision revisions or **Start again** for the
   newest page. Return links preserve the collection member page/status,
   collection-list cursor, catalog filters, and mounted application prefix.
   The JSON link reads the same selection again; it is not a frozen download.

No browser control submits or changes a decision. Revise opinions through the
existing explicit screening interface, using the latest decision revision for
optimistic concurrency. A successful repeat of the same opinion with the new
expected revision is a new decision record. Repeating an old expected revision
fails with 409 and appends nothing.

## Honest legacy and membership boundaries

Application startup creates only the additive history table; **no old decision
is copied into it**, and no old timestamp, reason, or revision is reconstructed.

| Stored situation | History response |
| --- | --- |
| Never screened | Empty `items`, `latest_review: null`, no first retained revision/time, zero unretained revisions, `current_status: unscreened`. |
| Legacy latest-only decision at revision N | Empty `items`, unchanged legacy `latest_review`, no first retained revision/time, `unretained_decision_revisions: N`. |
| First new submission after legacy revision N | First retained revision N+1 with its original timestamp; N earlier decision revisions remain unretained. |
| All decisions made with history enabled | First retained revision 1; zero earlier unretained revisions. |

The boundary describes **what this store retains**, not installation time or
proof of every real-world judgment. A legacy reason remains readable as the
latest opinion until updated; after that update, the overwritten legacy opinion
is **not** retrospectively preserved. Save it separately first if needed.

Any collection replacement, including a rename, changes its revision. Older
links fail with 409; use the collection list to open the new revision. Existing
opinions become `stale`, while their original collection revisions remain in
history. A removed member cannot be inspected through this reader (422).
Re-adding it reveals retained history again, but the old opinion stays stale,
not a current include. Resubmission uses the existing latest decision revision.
Deleted collections return 404; deletion does not erase decisions, history, or
papers. There is no purge, retention scheduler, or storage-size cap.

Changing paper contents under the same ID does not change a collection revision
or automatically stale a decision. History is not a versioned corpus, a signed
audit log, authenticated reviewer attribution, or scientific validation.

## API contract

`GET /collections/{collection_id}/screening/history`

| Query field | Contract |
| --- | --- |
| `document_id` | Required exact current member ID: 1-128 valid-Unicode characters, no surrounding whitespace. Embedded slashes, punctuation, NUL, CR/LF and combining characters remain exact. |
| `collection_revision` | Required positive integer, at most 9223372036854775807. Must equal the current stored revision. |
| `limit` | Integer 1-100, default 20. |
| `cursor` | Optional positive **decision revision** returned as `next_cursor`; exclusive, selects older decisions for this member. Must identify a retained revision for the requested member. |

Unknown or repeated fields, malformed UTF-8, empty cursors, float/boolean integer
spellings and query strings over 16,384 bytes fail with sanitized 422 responses.
Selectors are query-encoded, not appended to `/{document_id:path}`. An ID named
`history` or containing `/history` remains a literal document ID.

```bash
COLLECTION_ID=col_REPLACE_WITH_32_LOWERCASE_HEX_DIGITS
DOCUMENT_ID='replace/with?an-exact-member-id'
REVISION=1
curl --fail-with-body --silent --show-error --get \
  "http://127.0.0.1:8000/collections/$COLLECTION_ID/screening/history" \
  --data-urlencode "document_id=$DOCUMENT_ID" \
  --data-urlencode "collection_revision=$REVISION" \
  --data-urlencode "limit=20"
```

Use actual values from the collection API. Shell variables cannot contain NUL;
use a client library's query parameters for those identities. There is no ID
normalization, case-folding, or guessed substitute.

The versioned `ScreeningHistory` response contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1.0` for this history envelope. |
| `collection_id`, `collection_name`, `collection_revision`, `document_id` | Exact selection and current collection metadata from this read. |
| `current_status` | `unscreened`, `include`, `exclude`, `unsure`, or `stale`; not derived from a historical page row. |
| `latest_review` | Existing unchanged `ScreeningReview` shape, or null. May be a legacy latest-only decision. |
| `first_retained_revision`, `first_retained_at` | Earliest retained decision revision and its original `updated_at`; null if none exists. |
| `unretained_decision_revisions` | Revisions before retained history begins, or the legacy latest revision when no history exists; not a count of unrecorded real-world actions. |
| `items` | Bounded retained `ScreeningReview` records, descending by decision revision. Reasons and recorded `updated_at` values are preserved, never regenerated. |
| `next_cursor` | Last returned decision revision if older records remain, otherwise null. |

A `ScreeningReview` contains `schema_version`, exact collection/document IDs,
`collection_revision`, decision `revision`, `decision`, `reason`, and `updated_at`.
`updated_at` is the original successful submission timestamp for each history
row, not the time the history was read. Revision ordering is deterministic even
when timestamps tie or the system clock moves backward.

Each request resolves membership/existence, validates the requested collection
revision, reads latest metadata and history, and builds the result within one
SQLite **read-only transaction**. Different pages are not one frozen snapshot:
new submissions can change `latest_review` and `current_status` between requests.
Descending exclusive cursors avoid repeating newly appended records on older
pages; start again to see new decisions. Cursors are not authorization tokens.

## Python: the same bounded reader

```python
from pathlib import Path
from storage.paper_screening import SQLitePaperScreening

screening = SQLitePaperScreening(Path("/absolute/path/to/existing/corpus.sqlite3"))
collection = screening.collections.get("col_REPLACE_WITH_32_LOWERCASE_HEX_DIGITS")
document_id = collection.document_ids[0]
page = screening.list_history(
    collection.collection_id,
    document_id,
    collection_revision=collection.revision,
    limit=20,
)
print(page.current_status, page.first_retained_revision, page.unretained_decision_revisions)
for opinion in page.items:
    print(opinion.revision, opinion.collection_revision, opinion.updated_at, opinion.reason)
if page.next_cursor is not None:
    older = screening.list_history(
        collection.collection_id,
        document_id,
        collection_revision=collection.revision,
        limit=20,
        cursor=page.next_cursor,
    )
```

As before, service construction may initialize schema. **The history read itself**
opens the existing database with `mode=ro`; it creates no schema, repairs no data,
and cannot write even while a concurrent writer changes the corpus. Direct
Python arguments are strictly validated without numeric coercion; failures use
sanitized `CollectionError` codes/statuses. This reader does not call retrieval,
reranking, fake/live generation, network providers, or the agent event log.

## Bounds, failures, and privacy

History SQL loads at most `limit + 1` rows for pagination plus bounded
first/newest/cursor lookups. Each stored JSON record is capped at 16,384 characters
(a 16,385-character prefix detects overflow). Latest-decision validation retains
the existing maximum of 100 collection members. The response, including latest
and boundary metadata, must fit **524,288 UTF-8 bytes**. Reasons remain nonblank
valid Unicode, at most 1000 characters, without NUL. Reasons are never truncated.

`GET /explore/screening-history` adds `collection_id` and optional validated
`collection_return` navigation to the same selection/page fields. It allows
65,536 query bytes for encoded return state and caps complete escaped HTML at
1 MiB. Return state must match the requested collection ID/revision and cannot
be an external redirect. There are no scripts, external resources or browser
storage; escaping, hash-only style CSP, frame denial, no-referrer, no-store and
nosniff reuse the existing explorer helpers.

| HTTP status | Meaning |
| --- | --- |
| 404 | Exact collection is missing/deleted. |
| 409 | Collection revision changed, a saved corpus member is missing, or saved latest/history data is invalid or inconsistent. |
| 413 | Complete history JSON or escaped HTML exceeds its byte budget. Request a smaller page; no partial success is returned. |
| 422 | Invalid query, member, cursor, revision or navigation. Nothing was silently widened. |
| 405 | The native history page accepts GET only. |
| 503 | Existing storage is unavailable; this is not empty history. |

Selected history rows, lookahead, boundary and latest records are validated,
including revision gaps encountered while paging; the reader does not perform
an unbounded whole-history integrity scan. Direct database administrators can
alter SQLite, so application append-only behavior is not tamper evidence.

No authentication, per-user isolation or author identity is added. Keep the
service on loopback or behind your own access controls. History now retains past
reasons even after a newer opinion replaces the latest one: consider this before
recording sensitive information. IDs/return state in URLs can remain in browser
history and infrastructure logs. No-store is not access control or erasure.

## Reproduce the executed offline illustration

With the existing development dependencies available, choose unused output paths:

```bash
uv run --no-sync python -m scripts.demo_screening_history \
  --output-dir /tmp/scholar-screening-history-demo
uv run --no-sync python -m scripts.create_screening_history_gif \
  --transcript /tmp/scholar-screening-history-demo/transcript.txt \
  --output /tmp/scholar-screening-history.gif
```

The demo saves actual JSON and native HTML responses, measured checks and a
four-panel transcript. It uses only original synthetic notes and an explicitly
seeded pre-history-style latest row; setup writes occur before guarded reads.
It executes include/exclude/unsure submissions, rejected retry, the legacy
boundary, a collection revision change, exact history/return links, pagination,
Python/HTTP agreement, original-record equality and byte-identical restart HTML.
Reads measure unchanged database bytes, zero agent events and zero forbidden
calls. The temporary corpus is removed; outputs are never overwritten.

The committed GIF and [transcript](../assets/screening-history.txt) use that
executed output. Tests also exercise transaction rollback, concurrent writers,
legacy/restart behavior, stale/removal/re-addition semantics, response/query
bounds, exact IDs, HTML escaping, mounted prefixes, diagnostics and compatibility.
The GIF is an output illustration, not fabricated browser frames.

## Inspiration and portfolio scope

The **2026-10-15 America/Los_Angeles** workflow review used
[ASReview LAB's screening documentation](https://asreview.readthedocs.io/en/latest/lab/screening.html)
for revisable human relevance judgments and transparent, reproducible review,
and [Zotero's collections documentation](https://www.zotero.org/support/collections_and_tags)
for non-destructive organization. Coordinator-observed GitHub stars at that
review were approximately 1039 for ASReview and 15,612 for Zotero; these are dated
context, not measures of research quality.

Our original feature is an append-only local human-decision history with
bounded Python/HTTP/native browser readers. It does **not** implement ASReview
active learning, Zotero synchronization/reference-management parity, automatic
ranking, authenticated approvals or PRISMA compliance. No code, text or media
from either project is copied. Runtime model defaults are unchanged.

For a portfolio, demonstrate the atomic write/rollback contract, one explicit
legacy boundary, the latest-versus-historical distinction, bounded snapshot
pagination and safe native navigation. Describe the actual stack: SQLite,
Pydantic, FastAPI, HTTPX tests, and the existing custom agent state machine,
not a new model or front-end framework.

Related: [screening](PAPER_SCREENING_GUIDE.md), [collection browser](COLLECTION_BROWSER_GUIDE.md),
[latest-only exports](SCREENING_EXPORT_GUIDE.md), [architecture](../../ARCHITECTURE.md),
[safety](../../SAFETY.md), and [configuration](../../CONFIGURATION.md).
