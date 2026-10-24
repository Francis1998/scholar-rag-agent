# Save an auditable human screening progress report

Turn one complete [screening export](SCREENING_EXPORT_GUIDE.md) into a saved
JSON or literal Markdown report: exact collection ID/name/revision, every member
and status, resolved versus still-needing-human-review totals, and exclusion
reasons with exact member IDs. This is a **human progress handoff**, not automatic
screening, a PRISMA flow diagram, study-quality scoring, recall estimation or a
stopping recommendation.

[![Executed synthetic human screening report preview](../assets/screening-progress-report.png)](../assets/screening-progress-report.gif)

The static preview links to an opt-in four-frame GIF. It is a **generated
illustration of actual offline synthetic execution**, not browser footage,
screenshots, live inference or a scientific benchmark. The
[measured transcript](../assets/screening-progress-report.txt) is the accessible
alternative. The demo uses nine original synthetic notes: five resolved,
four still needing review, and four current exclusions in three exact-string
reason groups. New runs measure their own bytes, timestamps, IDs and hashes.

## Why this closes a workflow gap

This repository already saved decisions and history, displayed collection
counts, and exported complete JSON/CSV rows. It lacked a portable **progress
report** connecting those counts to the exact excluded members and their
original reasons, while keeping stale opinions out of current decisions.
The report is a pure derivation from that existing complete export, not another
connector, cue extractor, table, history scan or alternate comparison browser.

Workflow references checked on **2026-10-23 America/Los_Angeles**:

| Reference | Relevant workflow and deliberate differences |
| --- | --- |
| [ASReview LAB](https://github.com/asreview/asreview) | Transparent human screening, saved/editable labels, progress dashboards and export motivate an inspectable handoff. This feature does not implement ASReview's active learning, validation, prioritization, dashboard or project format. |
| [PaperQA](https://github.com/Future-House/paper-qa) | Inspectable cited evidence in a literature workflow motivates preserving exact paper identities and human reasons. This report performs no search, evidence synthesis, citation inference or autonomous review. |

No peer code, graphics, logos or performance results are reused. The surrounding
runtime remains FastAPI, Pydantic, HTTPX and SQLite with a custom ODA state
machine, not LangGraph. Default retrieval uses lexical hash vectors/BM25/RRF,
not learned semantic embeddings. **This feature does not run retrieval or models.**

## Browser: discover, download, inspect

1. Run the [isolated loopback API](../../QUICKSTART.md). Ingest authorized notes
   and save a [collection](PAPER_COLLECTIONS_GUIDE.md); optionally record
   [human include/exclude/unsure decisions](PAPER_SCREENING_GUIDE.md).
2. Open `/explore`, choose **Browse saved collections**, then **Inspect membership
   and human screening** for the intended collection.
3. Under **Revision-bound downloads**, choose **Human screening progress JSON**
   or **Human screening progress Markdown**. Native links include the current
   collection revision and respect a mounted application's `root_path`.
4. Open the saved Markdown in a text/Markdown reader. Use JSON for exact
   programmatic inspection. Every reason group identifies its members, and
   the full original snapshot is included for auditing statuses and totals.

Visible collection paging and status filters **do not filter the report**.
There is no write form, JavaScript, new browser page or hidden action. Paper
links in the existing collection detail use query-encoded exact document IDs,
not guessed path suffixes. Download filenames use only the validated collection
ID and numeric revision; untrusted names never enter headers.

## Semantics and ordering

| Field or total | Exact definition |
| --- | --- |
| `snapshot.total_documents` | Every current saved member, including distinct IDs with identical labels; no deduplication |
| `snapshot.counts` | Disjoint `include`, `exclude`, `unsure`, `unscreened`, `stale` counts, whose sum is the total |
| `resolved_documents` | Current `include` plus current `exclude`; not final acceptance, truth or quality |
| `needs_review_documents` | `unscreened` plus `unsure` plus `stale` |
| `resolved_document_ids`, `needs_review_document_ids` | Disjoint, complete partitions of members in saved collection order |
| `exclusion_reasons` | Only members whose **current status** is `exclude`; each group has original `reason`, `count` and exact `document_ids` |
| `snapshot.included_document_ids` | Only current includes; stale includes are never activated |
| `snapshot_sha256` | SHA-256 of the complete source export's UTF-8 bytes: `snapshot.model_dump_json(indent=2) + "\n"` |

Reasons group by **exact Python/Unicode string equality**. Case, whitespace,
multiline text and normalization forms are not folded, trimmed or merged.
Identical strings share a group; groups follow the first current exclusion in
saved membership order, and IDs within a group retain that same order.
The sum of reason-group counts equals the current exclude count. Every
excluded member appears exactly once, even when another member has identical
title/source labels.

An old exclude remains visible in `snapshot.items[].review` but contributes
only to `stale`, never to an exclusion group. An old include is likewise not
resolved. Any collection revision change, **including a rename**, makes old
opinions stale. Removed members are absent; re-added members require explicit
re-screening before an old opinion becomes current.

### Snapshot identity is not a collection-wide review version

The existing `SQLitePaperScreening.export_results(..., format="json")` runs
once per report request. It validates and reads all membership, current labels
and latest reviews in one bounded SQLite transaction. Serialization and grouping
use only those copied values; no second collection, queue or history read occurs.

Collection revisions bind name/membership, **not later decision edits or corpus
contents**. Two downloads at the same revision can legitimately differ after
someone records a new opinion or edits a label. The digest identifies the exact
exported values, not database state, paper contents, reviewer identity or a
cryptographic signature. Save one JSON report and derive its Markdown offline
if you need both formats from exactly the same read. No export-time timestamp,
new event or generated opinion is invented; original review timestamps and
decision revisions remain in the nested snapshot.

## API

Use the actual returned collection ID and revision; a placeholder is not a valid ID.
For a mounted application, include its prefix in `BASE_URL`.

```bash
BASE_URL='http://127.0.0.1:8000'
COLLECTION_ID='replace-with-your-col_id'
REVISION=2
OUTPUT="$(mktemp -d)"
curl --fail --silent --show-error --get \
  "$BASE_URL/collections/$COLLECTION_ID/screening-report" \
  --data-urlencode "collection_revision=$REVISION" \
  --data-urlencode "format=json" --output "$OUTPUT/report.json"
curl --fail --silent --show-error --get \
  "$BASE_URL/collections/$COLLECTION_ID/screening-report" \
  --data-urlencode "collection_revision=$REVISION" \
  --data-urlencode "format=markdown" --output "$OUTPUT/report.md"
```

`GET /collections/{collection_id}/screening-report` requires
`collection_revision`, an ASCII unsigned decimal integer of at most 19 digits
and value 1 through 9,223,372,036,854,775,807. `format` defaults to lowercase
`json`; the only other value is `markdown`. Signs, decimal points, whitespace,
non-ASCII digits, duplicate or unknown query parameters, invalid UTF-8/NUL
and query strings over 16,384 bytes fail. There is no `limit`, cursor, status
filter, history selector or partial export.

Only GET is accepted. The route deliberately sits outside the existing
`/screening/{document_id:path}` PUT wildcard, so PUT to a report URL cannot
record a decision for a paper named `report`.

Success uses `Content-Disposition: attachment` with
`screening-report-col_<32-hex-digits>-rN.json` or `.md`,
`Cache-Control: no-store`, and `X-Content-Type-Options: nosniff`.
Media types are `application/json` and `text/markdown; charset=utf-8`.
Only consume a file after the transfer succeeds and parsing completes; an
interrupted network transfer is not a complete artifact.

### JSON and Markdown contracts

Schema `1.0` contains the unchanged complete `ScreeningExport` as `snapshot`,
the digest and derived fields above, plus explicit `resolved_statuses`,
`needs_review_statuses` and `reason_grouping="exact_string_first_occurrence"`.
JSON is indented UTF-8, without a BOM and with a trailing newline.
IDs/reasons remain exact. No document body, arbitrary metadata, reviewer
identity, full history, inferred stage or model output is added.

Markdown has a fixed status table, the two auditable ID partitions, every
current exclusion reason and member group, and the entire source snapshot.
Dynamic names/IDs/snapshot values stay in JSON fences, and original reason text
stays in a literal fence longer than any embedded backtick run. HTML and
Markdown-looking reasons never become executable markup. Control characters in
the audit JSON are escaped; original CR/LF and normalization forms remain
recoverable even if a Markdown viewer displays them differently.

The existing title/source prefixes (300/512 characters) and **both original
truncation flags** remain unchanged. This does not pretend those labels are
full bibliographic records. Reports never truncate members or human reasons.
Existing opaque document IDs may contain NUL; JSON escapes it losslessly.
Names, title/source labels and reasons cannot contain NUL. Invalid Unicode
fails, not replacement-character repair.

## Python: complete standalone offline example

Save this example as a Python script and pass a **new absolute output directory**
as its first argument. It writes only its temporary synthetic database and the
two explicitly requested report files; no API server or credentials are needed.

```python
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from retrieval.models import Document
from storage.document_store import SQLiteDocumentStore
from storage.paper_screening import ScreeningSubmission, SQLitePaperScreening
from storage.screening_report import ScreeningReport, export_screening_report

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=False)
with TemporaryDirectory(prefix="screening-report-example-") as temporary:
    database = Path(temporary) / "synthetic.sqlite3"
    SQLiteDocumentStore(database).add_documents(
        [
            Document(
                document_id=identifier,
                title="Original synthetic note",
                source="synthetic:guide",
                text="Not a publication or research finding.",
            )
            for identifier in ("synthetic-outside", "synthetic-unread")
        ],
        [],
    )
    screening = SQLitePaperScreening(database)
    collection = screening.collections.create(
        name="Synthetic human review",
        document_ids=["synthetic-outside", "synthetic-unread"],
    )
    screening.submit(
        collection.collection_id,
        "synthetic-outside",
        ScreeningSubmission(
            collection_revision=collection.revision,
            expected_decision_revision=0,
            decision="exclude",
            reason="Outside the synthetic scope.\nHuman judgment, not a quality verdict.",
        ),
    )
    download = export_screening_report(
        screening,
        collection.collection_id,
        collection_revision=collection.revision,
    )
    with (output / "report.json").open("xb") as destination:
        destination.write(download.content)

# This same-snapshot Markdown derivation works after the database is removed.
report = ScreeningReport.model_validate_json(download.content, strict=True)
with (output / "report.md").open("xb") as destination:
    destination.write(report.to_markdown().encode("utf-8"))
print(report.resolved_documents, report.needs_review_documents)
print(report.exclusion_reasons[0].document_ids)
print(output.resolve())
```

Measured expectations: `1 1`, followed by `('synthetic-outside',)` and the output
directory. The guide example is executed by the test suite, not just linted.

For an existing application, pass its `container.paper_screening`, actual
collection ID and current revision to `export_screening_report`. It returns
the existing immutable `ScreeningDownload` shape with checked `content` bytes,
`media_type`, and safe `filename`. It adds no store, migration or settings.
Constructing `SQLitePaperScreening` separately retains its existing startup
schema-initialization behavior; report reads themselves perform no writes.

For a previously obtained **complete** `ScreeningExport`, call
`build_screening_report(snapshot)` directly; this pure function does no I/O.
It revalidates fields, provenance, unique exact membership, count sums/statuses
and inclusion ordering, including unchecked `model_copy` inputs.
Do not pass a queue page or manually stitched pages. The report's serialization
also revalidates its summaries; tampered totals/group membership fail rather
than being silently recomputed into a success.

## Limits, failures and privacy

| Fixed bound | Behavior |
| --- | --- |
| Complete source export | 1-100 members and at most 524,288 serialized UTF-8 bytes; existing bounded SQLite readers/validation reused |
| IDs/reasons | Existing exact 128-character document IDs, nonblank 1-1000-character reasons, signed-64-bit positive revisions |
| Report body | Each requested JSON/Markdown serialization must fit 1,048,576 UTF-8 bytes, including indentation, escaping, repeated reasons and audit snapshot |
| API query | 16,384 raw bytes; only one revision and one optional format |

The source JSON must fit even when requesting Markdown. A report's requested
format can fail its own cap even when another fits. There is no page-size
workaround or truncation; headers are not part of the byte caps.

| HTTP | Stable code | Meaning |
| --- | --- | --- |
| 404 | `collection_not_found` | Exact collection is absent; no replacement is guessed |
| 405 | `screening_report_method_not_allowed` | Only GET; no write occurred |
| 409 | `collection_revision_conflict` | Deliberately reopen the current revision |
| 409 | `collection_documents_missing` | Restore missing documents or explicitly revise membership |
| 409 | `invalid_collection_record`, `invalid_screening_record`, `invalid_document_record` | Source data is invalid; no repair, skipped member or empty-success fallback |
| 409 | `invalid_screening_snapshot`, `invalid_screening_report` | Python snapshot/report consistency check failed |
| 413 | `screening_export_too_large`, `screening_report_too_large` | Complete serialization cannot fit; no partial download |
| 422 | `invalid_screening_report_request` | Correct IDs, bounds, Unicode, duplicates or unknown parameters |
| 503 | `collection_storage_error` | Existing storage is unavailable, not an empty collection |

Invalid Python selector/format inputs raise Pydantic `ValidationError`; numeric
strings, booleans and floats are not Python revisions. Storage/serialization
failures use `CollectionError`. Malformed portable models can also raise
`ValidationError` while loading. API errors preserve collection no-store/nosniff
headers and omit attachment headers. Logs contain stable codes, not private
reasons or raw database diagnostics.

No model/live-or-fake generation, network, retrieval, history scan, collection
mutation or event write occurs during reporting. The offline demo measures
forbidden-call counts and unchanged database bytes. No report is automatically
saved server-side: the user explicitly saves a download.

Review sensitive names/labels/reasons/IDs before sharing. `no-store` does not
encrypt or delete a file. This local/trusted API has no multi-user authentication,
tenant isolation or authenticated reviewer attribution; keep it on loopback or
behind independent access controls. Do not interpret a fully resolved collection
as a complete literature search or complete scientific review.

## Reproduce the demo and portfolio illustration

With Python 3.11+ and the existing development dependencies installed:

```bash
OUTPUT_PARENT="$(mktemp -d)"
OUTPUT="$OUTPUT_PARENT/screening-report"
uv run --no-sync python scripts/demo_screening_report.py --output-dir "$OUTPUT"
uv run --no-sync python scripts/create_screening_report_gif.py \
  --transcript "$OUTPUT/transcript.txt" \
  --output "$OUTPUT/preview.gif" --still "$OUTPUT/preview.png"
printf 'Actual artifacts: %s\n' "$OUTPUT"
```

The demo saves actual catalog/list/detail HTML, the JSON and Markdown report,
its complete source export, the stale-revision error, `checks.json` and the
four-panel transcript. The GIF/still use the existing small Pillow renderer.
Existing output directories/files and symlinks are refused. Synthetic setup
writes are explicit; report reads are guarded against networking (including
sockets), generation, retrieval, history scans and mutations. Ambient keys,
dotenv settings and database paths are ignored.

The demo filters the visible collection to one unread member, downloads all
nine, audits exclusion groups `[2, 1, 1]`, observes HTTP 409 for a stale
revision, verifies restart-identical report bytes, and regenerates Markdown
from saved JSON after removing the temporary database. It records actual byte
lengths, SHA-256 values, statuses and forbidden-call counts. Random collection
IDs and original submission timestamps vary between fresh demos; unchanged
saved data is deterministic across reads/restarts.

**Portfolio use case:** hand a collaborator the saved JSON/Markdown and point
from an exclusion total to the exact member IDs and verbatim human reasons.
Show why the two stale opinions still need review, how case/whitespace-distinct
reasons stay separate, and why a visible page is not the export scope. Present
the transcript and source artifact beside the opt-in animation. Claim these
engineering guarantees only, not scientific quality or screening performance.

The [provider catalog check](PROVIDER_MODELS_GUIDE.md) remains documentation
only: no selected model ID, endpoint, budget or compatibility date is changed
by this zero-model feature.
