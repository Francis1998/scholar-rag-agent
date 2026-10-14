# Inspect duplicate candidates in a saved paper collection

Inspect a **bounded existing selection** before screening or retrieval. The
Python service and HTTP endpoint reuse `DuplicatePaperClusterer` against current
stored document metadata, returning groups, singletons, exact IDs and warnings.
They never choose survivors, delete or hide papers, change labels, or make a
scientific same-work judgment.

![Measured synthetic collection duplicate inspection](../assets/collection-duplicates.gif)

This four-panel GIF is generated from actual guarded offline API/Python output.
It is an illustration, not a browser recording, duplicate-recall benchmark, or
scientific validation. The [reproduction](#reproduce-the-offline-demo-and-gif)
saves complete machine-readable artifacts; the panels only summarize them.
The [committed transcript](../assets/collection-duplicates.txt) is the measured
input used to render the showcased GIF.

## 1. Create a collection using real document IDs

Start the local API using the isolated offline instructions in the
[Quickstart](../../QUICKSTART.md). Keep the database path unchanged across
restarts. In `/explore`, copy exact document IDs, or read
`GET /documents?limit=20` and follow `next_cursor`. Catalog labels may be
truncated for browsing; this inspection reads full bounded labels and refuses
oversize inputs rather than matching clipped titles.

For a fresh synthetic example with the server running:

```python
import httpx

base = "http://127.0.0.1:8000"
with httpx.Client(base_url=base, timeout=30) as client:
    ids = []
    for title in ("Synthetic GraphRAG method note", "Synthetic GraphRAG method notes"):
        response = client.post("/ingest/text", json={
            "title": title,
            "text": "Synthetic data only. GraphRAG connects evidence; review it yourself.",
            "source": "synthetic:duplicate-walkthrough",
        })
        response.raise_for_status()
        ids.append(response.json()["document_id"])
    response = client.post("/collections", json={
        "name": "Synthetic candidate review",
        "document_ids": ids,
    })
    response.raise_for_status()
    saved = response.json()
    print(saved)  # Copy the actual collection_id and revision, not the name.
    response = client.get(f"/collections/{saved['collection_id']}/duplicates", params={
        "collection_revision": saved["revision"],
        "title_threshold": 0.92,
        "format": "json",
    })
    response.raise_for_status()
    print(response.text)
```

The fixture deliberately keeps similar titles with distinct text-ingestion
identities. For existing selections, `GET /collections` and
`GET /collections/{collection_id}` recover real IDs and the current revision.
Do not infer a collection ID from its name or assume revision 1 after editing.
The [collection guide](PAPER_COLLECTIONS_GUIDE.md) covers creation, replacement,
conflicts and exact membership.

## 2. Inspect through HTTP or standalone Python

```text
GET /collections/{collection_id}/duplicates
    ?collection_revision=N
    &title_threshold=0.92
    &format=json
```

`collection_revision` is mandatory. `title_threshold` defaults to `0.92`, accepts
finite numbers from 0 through 1 inclusive, and uses the clusterer's inclusive
title-match threshold. Zero can group unrelated nonblank titles; one requires
equal normalized titles, not equal papers. `format` is `json` (default) or
`markdown`. Unknown or repeated query parameters are rejected, not ignored;
there is no unscoped, filtered, paginated or partial-report mode.

The standalone Python reader does **not** initialize schemas, build retrieval
indexes, load settings or configure models:

```python
from pathlib import Path
from storage.collection_duplicates import SQLiteCollectionDuplicates
from storage.paper_collections import CollectionError

database = Path("/absolute/path/to/your/existing-corpus.sqlite3")
service = SQLiteCollectionDuplicates(database)
try:
    report = service.inspect(
        "col_replace_with_the_actual_32_hex_digits",
        collection_revision=1,  # Replace with the actual saved revision.
        title_threshold=0.92,
    )
except CollectionError as error:
    print(error.status_code, error.code, str(error))  # Sanitized, no stored text.
    raise
print(report.inspected_count)
for group in report.groups:
    print(group.cluster_id, group.match_reasons)
    for member in group.members:
        print(member.document_id, member.title, member.source, member.doi, member.doi_field)
print("Unclustered:", [member.document_id for member in report.singletons])
```

Replace the illustrative database/ID/revision before running. Python accepts
strict integer revisions (not bools or numeric strings) and finite numeric
thresholds (not bools or numeric strings). `AppContainer.collection_duplicates`
exposes the same reader in applications already using a container. Normal
container startup still initializes the existing stores/retrieval index;
inspection requests themselves do not.

Use `report.to_json()` and `report.to_markdown()` for the exact HTTP body
representations. Save privately with exclusive creation, for example
`with Path("/absolute/new/path/candidates.json").open("x", encoding="utf-8") as
output: output.write(report.to_json())`; do not overwrite an earlier inspection
unless that is intentional.

## 3. Interpret the complete report

| Field | Meaning |
| --- | --- |
| `schema_version` | Report format `"1.0"`; no new stored schema |
| `collection_id`, `collection_name`, `collection_revision` | Actual collection metadata from the same read snapshot as its members |
| `title_threshold`, `inspected_count`, `document_ids` | Effective threshold, complete count, exact IDs in ascending case-sensitive order |
| `groups` | Candidate groups with a report-local `cluster_id`, full `members`, and cluster-building `match_reasons` |
| `singletons` | Every member not grouped with another selected paper, with the same full member fields |
| `warnings` | Heuristic, privacy, transitivity, missing-signal and current-corpus limitations |

Every selected ID occurs exactly once among groups and singletons. Immutable
Python models contain tuples, not caller-owned mutable membership lists.
Group ordering follows the existing helper: descending size, then cluster ID.
Members follow sorted exact input IDs; singletons follow that order too.
Group IDs are positional labels, **not** durable paper identities or canonical
survivors. Adding a member may renumber them.

Each member contains `document_id`, `title`, `source`, the chosen **raw** `doi`
string and `doi_field` (`doi`, `paper_doi`, `work_doi`, or null). No title,
source, ID or DOI is clipped. Document bodies, chunks, screening decisions and
unrelated metadata values are not returned.

Example member from the actual synthetic demo:

```json
{
  "document_id": "doi-a",
  "title": "Synthetic DOI fixture",
  "source": "synthetic:duplicate-inspection",
  "doi": " DOI:10.0000/note ",
  "doi_field": "doi"
}
```

The demo groups `doi-a` with `doi-b` with reason `doi`, and `title-a` with
`title-b` with reason `title:0.984`. `Paper`, `paper`, and `solo` are separate
singletons. All seven selected identities remain present; the eighth stored,
unselected document is absent. These are fixture results, not real publications.

The helper selects the first nonblank DOI string in `doi`, `paper_doi`,
`work_doi` order, lowercases it and trims recognized `doi:`, `https://doi.org/`,
or `http://doi.org/` prefixes and outer whitespace/periods for matching. The
report retains the original chosen string and its field. Lower-priority aliases
are validated but not combined or treated as additional signals. No DOI syntax,
registration, landing page or bibliographic truth is externally checked. An
absent/blank/normalization-empty string is an absent signal; malformed stored
metadata instead fails with 409.

Titles are whitespace-normalized and lowercased for `SequenceMatcher`.
Blank titles never fuzzy-match, including at threshold zero. Distinct
case-sensitive raw document IDs remain distinct unless their DOI/title signal
links them. The reused union-find may form **transitive** groups and may link
conflicting DOI values via similar titles or conflicting titles via DOI.
Reasons summarize only successful cluster-building links, rounded to three
decimal places for title ratios; they are not exhaustive pairwise evidence.
Singleton status is not proof of uniqueness. No comprehensive recall is claimed.

Markdown has a fixed heading and the **entire JSON report in a dynamically
fenced literal block**. This deliberately preserves the complete payload rather
than making unsafe source links or interpolating untrusted HTML. Embedded
backticks, links, newlines and HTML remain data; JSON escaping is reversible.

## 4. Make a human selection, then preview or screen

Read the groups and the underlying papers. Copy a **nonempty list of exact
document IDs you chose**; do not blindly select one member per group. For the
synthetic demo only, the explicit illustrative choice is `["doi-a", "title-a"]`:

```json
{"query":"What does GraphRAG connect?","document_ids":["doi-a","title-a"]}
```

Send this to `POST /retrieve` for generation-free evidence preview. For your own
corpus replace those IDs with actual chosen report IDs. Inspection itself never
calls retrieval. Do not send `collection_id` instead if you want a subset: it
still means **every** current collection member. Never replace an empty selection
with null or omit `document_ids`; that would widen the scope.

To record your own screening decision, use the existing revision-checked
screening API. This example explicitly writes an `include` opinion only after
the caller has chosen a document and reason:

```python
from urllib.parse import quote
import httpx

collection_id = "replace-with-the-real-collection-id"
document_id = "replace-with-a-human-selected-document-id"
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30) as client:
    saved = client.get(f"/collections/{collection_id}")
    saved.raise_for_status()
    revision = saved.json()["revision"]
    queue = client.get(f"/collections/{collection_id}/screening", params={
        "collection_revision": revision, "limit": 100,
    })
    queue.raise_for_status()
    item = next(item for item in queue.json()["items"] if item["document_id"] == document_id)
    response = client.put(
        f"/collections/{collection_id}/screening/{quote(document_id, safe='')}",
        json={
            "collection_revision": revision,
            "expected_decision_revision": item["review"]["revision"] if item["review"] else 0,
            "decision": "include",
            "reason": "My explicit judgment after inspecting source content and review criteria.",
        },
    )
    response.raise_for_status()
```

Read the new queue and explicitly pass its nonempty `included_document_ids` to a
preview if that is your intended scope. Decisions are never inferred from the
duplicate report. The [screening guide](PAPER_SCREENING_GUIDE.md) explains stale
decisions, optimistic concurrency and manual include/exclude/unsure labels.

## Bounds, errors, and data handling

| Boundary | Complete limit |
| --- | --- |
| Selection | Existing saved collection, 1-100 unique IDs; exact validated IDs up to 128 characters |
| Revision | Required integer 1 through 9,223,372,036,854,775,807 |
| Title/source | At most 512 Unicode characters each; SQL checks at most 2,048 stored bytes each before decoding |
| Metadata | At most 16,384 **stored bytes** per document, checked in SQL before decoding/JSON parsing; JSON object with unique string keys and string values |
| DOI aliases | Each supplied `doi`, `paper_doi`, `work_doi` string at most 512 characters, even if lower-priority |
| JSON and Markdown | Each entire UTF-8 body at most 262,144 bytes (256 KiB), including warnings/formatting; **both must fit before either is returned** |

No body or chunk fields are read by the inspection service. It resolves
membership with `SQLitePaperCollections.resolve_in_snapshot` on a `sqlite3.Row`,
`mode=ro` connection and one `BEGIN` read transaction. Size/type probes and
bounded metadata reads share that snapshot. All selected rows are validated
before fuzzy matching; an invalid last member never produces a partial report.
Inputs exceeding a bound are rejected, not silently truncated into a match.
UTF-8 and UTF-16 SQLite databases are decoded using their actual encoding;
the metadata bound applies to stored bytes, not re-encoded JSON length.

| HTTP status | Stable code / next action |
| --- | --- |
| 404 | `collection_not_found`: rediscover the saved collection |
| 409 | `collection_revision_conflict`: read current membership/revision and review changes |
| 409 | `collection_documents_missing`: repair the saved selection explicitly |
| 409 | `invalid_collection_record` / `invalid_duplicate_document`: repair invalid persisted data; not a guessed singleton |
| 413 | `duplicate_input_too_large`: review oversized selected metadata/labels; no clipping |
| 413 | `duplicate_report_too_large`: choose a smaller explicit saved selection; nothing is omitted automatically |
| 422 | `invalid_duplicate_request`: correct required fields, bounds, format, unsupported or repeated parameters |
| 503 | `duplicate_storage_unavailable`: restore readable existing collection/document storage |

Errors use `{"detail":{"code":"...","message":"..."}}`; diagnostics and logs do
not echo raw private records or SQLite exception text. All successful,
validation-error and operational-error responses use `Cache-Control: no-store`
and `X-Content-Type-Options: nosniff`. Attachment names contain only the validated
collection ID and actual revision, never its name, title or source labels.

This remains a local/trusted API without authentication, tenant isolation or
scientific validation. Restrict network access; protect SQLite, exported reports,
backups and logs. Exact IDs, labels, names and DOI strings can be sensitive.
The endpoint makes no network, live/fake model, DOI-resolver, retrieval or event
calls. It changes no corpus, collection, screening or saved-run state. No new
schema, migration, dependency, model default or environment setting is needed.
For later generation only, consult the current
[provider guide](PROVIDER_MODELS_GUIDE.md); this feature uses **no model**.

## Restart, reproducibility, and portfolio walkthrough

Reuse the same database, collection ID, revision and threshold after restart.
Unchanged selected metadata gives byte-identical JSON and Markdown and stable
cluster order. A collection edit requires its new revision. A metadata edit
under an unchanged document ID can change candidates **without changing the
collection revision**: this is a current-corpus inspection, not frozen evidence
across requests, and it cannot detect every scientific distinction.

For a portfolio demonstration, show the exact saved selection, compare raw
DOI aliases with title-only candidates, account for all groups/singletons, show
the separate `Paper`/`paper` identities, then compare restart bytes. Explain
why the fixture's explicitly chosen preview IDs are a human action and why
none of this establishes scientific equality or measured duplicate recall.
Tests also commit a concurrent WAL writer after collection resolution and prove
that the in-flight report keeps old membership, revision and labels; the next
request observes the new state.

### Reproduce the offline demo and GIF

From the repository checkout, with the existing dev environment:

```bash
DEMO_DIR="$(mktemp -d)/collection-duplicates"
uv run --no-sync python -m scripts.demo_collection_duplicates --output-dir "$DEMO_DIR"
uv run --no-sync python -m scripts.create_collection_duplicates_gif \
  --transcript "$DEMO_DIR/transcript.txt" \
  --output "$DEMO_DIR/collection-duplicates.gif"
uv run --no-sync python -m json.tool "$DEMO_DIR/checks.json"
```

The demo uses environment-ignoring `offline_settings`, synthetic document/DOI
strings, the real collection API, actual duplicate API/Python service, a real
restart, and an explicitly scoped preview. HTTP transports and socket connects
are trapped throughout; generation, retrieval during inspection, store writes,
screening and event mutations are guarded. It checks database bytes, exact
response bytes/digests and selection identities, not just a successful status.
The temporary database is removed; only the requested new artifact files remain.

Artifacts include the real collection ID/revision, both complete report formats,
their restart copies, the explicit-ID preview, measured checks and transcript.
Names are exclusively created and collisions/symlinks fail without overwriting.
The four 1120-by-540 frames reuse the existing Pillow renderer. Rendering the
same transcript again to a new path produces identical GIF bytes. Separate
demo runs generate new real collection IDs and corresponding report digests;
they are not expected to have identical IDs.

The fixture measures seven inspected papers, two candidate groups, three
singletons, a preview containing exactly `doi-a` and `title-a`, zero network
calls, zero forbidden calls and zero agent events. Complete JSON is 3,373 bytes
and literal Markdown is 3,502 bytes in this fixture. Fixture setup writes its own
temporary corpus/collection; inspection and preview do not mutate them. The
committed illustration is not a model demonstration, production-readiness claim
or comparative quality benchmark.

## Workflow inspiration

[ASReview](https://github.com/ASReview/asreview) documents automatic duplicate
detection/hiding during screening and exports including hidden records.
[PaperQA](https://github.com/Future-House/paper-qa) emphasizes local scientific
evidence and metadata inspection. These are workflow inspirations, not borrowed
code or claims that either project lacks a feature. Here the useful addition is
an inspectable, persisted-selection workflow before screening/retrieval, with
explicit human selection and **no automatic paper loss**. The existing pure
[duplicate-cluster helper](DUPLICATE_PAPER_CLUSTER_GUIDE.md) remains available
independently; this guide covers its new persisted-corpus service integration.
