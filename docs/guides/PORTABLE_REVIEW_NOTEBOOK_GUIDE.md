# Export a portable human-review notebook

Save one completed run's frozen query, answer and evidence **together with all
retained answer reviews and exact quote annotations**, in JSON or literal-safe
Markdown. Both histories are complete at one read-only SQLite snapshot, not
just the pages currently visible in the browser. No model, network, retrieval,
current-corpus read, opinion write or event write runs during export.

[![Executed synthetic notebook-export illustration: a saved answer with 23 separate human reviews and 23 annotations](../assets/portable-review-notebook.png)](../assets/portable-review-notebook.gif)

[Play the four-frame GIF](../assets/portable-review-notebook.gif).
This is an **illustration of executed synthetic output using the existing
Pillow renderer**, not browser screenshots, a recording, a scientific finding
or a benchmark. The still preview makes animation opt-in. The
[plain-text transcript](../assets/portable-review-notebook.txt) provides the
same measured information without animation.

## Browser workflow

1. Start the [offline Quickstart](../../QUICKSTART.md) with the database that
   contains the completed run. Open `/explore/runs` and choose its
   **Human-review notebook**.
2. Inspect the saved answer, independent human histories and exact quote
   source links. To record opinions, use the existing
   [answer-review](ANSWER_REVIEWS_GUIDE.md) or
   [annotation](EVIDENCE_ANNOTATIONS_GUIDE.md) submission APIs/Python services;
   the browser has no write controls.
3. Choose **Download complete notebook (JSON)** or **(Markdown)**. These native,
   keyboard-accessible links ignore page sizes, cursors and catalog return
   filters. Their query-addressed run ID preserves slashes, dot segments,
   punctuation, Unicode composition and mount prefixes without rewriting IDs.
4. Open Markdown in a trusted local viewer, or inspect JSON with ordinary
   tools. Query, answer, full captured passages and complete opinion records
   remain available with the server stopped and current corpus removed.
   Original source paths/URLs are provenance data, not dependencies that must
   still exist.

The existing **browser pages remain independently paginated reads**. A download
is a new, single snapshot and can include opinions appended since that page
was displayed. Two separate downloads can differ if a new opinion is committed
between requests. There is no snapshot token, export timestamp, ongoing sync
or promise that all future opinions are already included.

## API contract

| Operation | Contract |
| --- | --- |
| `GET /runs/{run_id}/notebook` | New complete notebook download; default `format=json` |
| `GET /explore/run/notebook/export?run_id=...` | Same export with a query-addressed exact ID; used by native links, including IDs unsuitable for a URL path |
| `format` | Exactly `json` or `markdown`; optional once, lowercase |
| `run_id` | Exact string, 1-256 Unicode code points; no trimming, case folding or normalization |
| Selection | One completed run and both entire retained histories, without paging or filtering |
| Response | JSON: `application/json`; Markdown: `text/markdown; charset=utf-8` |
| Filename | `review-notebook-<first 16 hex characters of SHA-256(run_id UTF-8)>.json` or `.md` |
| Headers | Attachment disposition, `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` |

Unknown/duplicate parameters, page filters and malformed UTF-8 fail with 422
before storage reads. The raw query string is limited to 16,384 bytes.
Path-addressed malformed UTF-8 is rejected rather than interpreted as a
replacement-character run ID. GET is the only method; other methods receive
405 with `Allow: GET` and the same privacy headers.

Use the query-addressed endpoint for opaque IDs rather than placing untrusted
values into a path. With an existing local API and a new output directory:

```bash
BASE_URL=http://127.0.0.1:8000
RUN_ID='copy-the-exact-completed-run-id'
OUTPUT_DIR="$(mktemp -d)"
curl --fail-with-body --silent --show-error --get \
  "$BASE_URL/explore/run/notebook/export" \
  --data-urlencode "run_id=$RUN_ID" --data-urlencode 'format=json' \
  --output "$OUTPUT_DIR/notebook.json"
curl --fail-with-body --silent --show-error --get \
  "$BASE_URL/explore/run/notebook/export" \
  --data-urlencode "run_id=$RUN_ID" --data-urlencode 'format=markdown' \
  --output "$OUTPUT_DIR/notebook.md"
uv run --no-sync python -m json.tool "$OUTPUT_DIR/notebook.json"
```

Check each command's exit status. A non-200 body is a structured error, not a
notebook, even if a client saves it under an artifact filename.

### JSON and Markdown contents

The new outer `schema_version: "1.0"` object contains:

| Field | Meaning |
| --- | --- |
| `evidence` | The unchanged version-one `EvidenceBundle`: original query, agent/run IDs, DONE time, configuration, plan, answer, claims, citations, generation identity, exact request, full captured sources, ranks/scores/paths/metadata/digests and operational trace |
| `reviews` | All `SavedAnswerReview` records: original sequence, run ID, UUID, decision, exact comment, cited chunk IDs, schema and UTC timestamp |
| `annotations` | All `SavedEvidenceAnnotation` records: original sequence, run ID, UUID, schema, UTC timestamp, document/chunk IDs, source-text digest, exact half-open span, quote and note |
| `history_order` | `sequence_desc`: newest original sequence first independently within each history; gaps remain |
| `consistency` | `single_sqlite_read_transaction`: saved events and both histories were read together |
| `warnings` | Explicit human-opinion, snapshot, provenance and sharing limitations |

Histories are arrays, not paginated pages: there is no `next_cursor`. Empty
arrays explicitly mean no retained opinions in the snapshot. Sequences from
the two tables are **not a shared clock**; UUIDs are record/retry identities,
not authenticated people. Timestamp values use the existing schemas' UTC
serialization, not a new export time. No consensus, latest verdict, reviewer
identity or scientific endorsement is invented.

Reviews' cited chunk IDs resolve only against `evidence.snapshot.sources`.
Every annotation's document ID, chunk ID, digest, `start`, `end` and quote must
match that exact source. Offsets are Unicode code points, not UTF-8 byte offsets
or grapheme positions. Two equal quoted strings can select different
occurrences; no text search or normalization replaces the stored selector.

Markdown puts every dynamic value inside the existing literal fences, with a
fence longer than any embedded backtick sequence. It gives readable query,
answer, passages, comments and notes, plus complete literal JSON records for
provenance. Source URLs, Markdown-looking instructions and HTML stay data,
not active links or markup. JSON is the canonical machine-readable artifact;
Markdown viewers can visually normalize whitespace/control characters.

The underlying evidence export privacy contract is unchanged: the trace stops
at the first terminal event; snapshot/generation event references resolve
inside `evidence`; unknown event payloads are explicitly omitted with warnings.
This does not promise full raw diagnostic logs or whole papers beyond the
captured passages.

## Fixed bounds and failures

All maxima are inclusive, finite code constants, **not new settings**.

| Boundary | Maximum |
| --- | --- |
| Answer reviews for the selected run | 1,000 retained records |
| Annotations for the selected run | 1,000 retained records |
| Stored review text per row | 262,144 UTF-8 bytes, summed across schema/run/UUID/time/decision/comment/raw cited-ID JSON text |
| Stored annotation text per row | Existing 16,384 UTF-8 bytes across its text fields |
| Stored text across both complete histories | 4,194,304 UTF-8 bytes; includes raw JSON whitespace/escaping and original timestamp text, excludes integer columns |
| Frozen events | Existing 100 records, 1,048,576 bytes per payload, 8,388,608 total UTF-8 event text bytes; includes later events in the read preflight |
| Complete JSON body | 8,388,608 UTF-8 bytes, including whitespace and final newline |
| Complete Markdown body | 8,388,608 UTF-8 bytes, including fences and repeated readable text |

Both full representations must fit **before either succeeds**. SQL first reads
at most the count limit plus one bounded metadata row, checks storage types and
byte lengths before fetching text, then checks exact decoded UTF-8 sizes.
UTF-16 SQLite storage uses a conservative byte preflight followed by the same
exact UTF-8 accounting. The original 4,000-code-point review comment,
100-cited-ID and 1,000-code-point quote/note validators still apply.

| Status / code | Meaning |
| --- | --- |
| 404 `run_not_found` | No events for that exact run |
| 409 `run_incomplete`, `run_failed`, `snapshot_unavailable` | Not a complete captured run; no reconstruction from today's corpus |
| 409 `invalid_run_record`, `invalid_review_record`, `invalid_annotation_record` | Corrupt/unsupported records, invalid references or mismatched exact anchors; no skipped rows |
| 413 `notebook_export_too_large` | Any notebook read count/byte limit or either complete serialization limit exceeded |
| 422 `invalid_notebook_export_request` | Invalid exact selector, format or request encoding/parameters |
| 503 `notebook_storage_unavailable` | Missing/unreadable database or required table; not an empty history |

Failures are sanitized, logged by code and returned as `{"detail":{"code":...,
"message":...}}`, with no attachment disposition, partial success or raw
exception/input echo. A 413 cannot be bypassed with `limit` or by requesting
the smaller format. Existing paginated APIs remain useful for inspection, but
combining their pages is not this atomic notebook. No records are deleted,
shortened or archived to make an export fit.

The export caps are not retention policies, authentication or hard wall-clock
deadlines. The SQLite connection retains the existing five-second busy timeout.
Startup still initializes ordinary stores/indexes; these no-work guarantees
cover export reads, not normal app initialization or the demo's explicit setup.

## Direct Python, without app startup

The standalone reader opens only an existing `mode=ro` database. Construction
does not create tables or files and does not load provider settings or a corpus.
Capture once, then write both formats from the same object:

```python
from pathlib import Path

from storage.review_notebook import SQLiteReviewNotebook

database = Path("/absolute/path/to/existing.sqlite3")
directory = Path("/absolute/path/to/new-notebook-download")
run_id = "copy-the-exact-completed-run-id"
notebook = SQLiteReviewNotebook(database).export(run_id)
directory.mkdir(parents=True)
for filename, content in (
    ("notebook.json", notebook.to_json()),
    ("notebook.md", notebook.to_markdown()),
):
    with (directory / filename).open("x", encoding="utf-8", newline="") as output:
        output.write(content)
for annotation in notebook.annotations:
    source = next(
        item
        for item in notebook.evidence.snapshot.sources
        if item.chunk.chunk_id == annotation.chunk_id
    )
    assert source.chunk.text[annotation.start : annotation.end] == annotation.quote
```

Handle `EvidenceExportError` (including `NotebookExportError`) using its safe
`code`, `status_code` and message. A trusted exported JSON artifact can later be
loaded with `ReviewNotebook.model_validate_json(...)` and rendered again without
the database. This is not an import/write API or authentication of an externally
edited file; hashes are integrity checks, not signatures.

## Reproduce the synthetic offline demo

Use Python 3.11+ and the existing development dependencies. No API keys,
downloaded papers, live provider tests or browser installation are needed.

```bash
uv sync --frozen --extra dev
DEMO_DIR="$(mktemp -d)/portable-review-notebook"
uv run --no-sync python -m scripts.demo_notebook_export --output-dir "$DEMO_DIR"
cat "$DEMO_DIR/checks.json"
uv run --no-sync python -m scripts.create_notebook_export_gif \
  --transcript "$DEMO_DIR/transcript.txt" \
  --output "$DEMO_DIR/portable-review-notebook.gif" \
  --still "$DEMO_DIR/portable-review-notebook.png"
```

The demo uses one real fake-adapter query over an original synthetic note,
23 human review submissions and 23 exact annotations. Default JSON pages show
20 of each; the browser can show one of each; native download links still
return all 23. It retains original sequences/UUIDs/timestamps and validates the
second occurrence of a repeated quote with its full-source digest.

Actual downloads are saved before the restart demonstration. The isolated
current corpus is changed, then deleted; both formats remain byte-identical
after app restarts. Saved events remain unchanged. Guards count forbidden
generation, retrieval, network, corpus/opinion/event writes during reads.
Finally the temporary database is removed and the JSON alone reproduces
the Markdown. The demo refuses existing output targets and ignores ambient
credentials/settings/databases.

Output: both notebooks, first-page review/annotation JSON, measured
`checks.json`, and a four-panel `transcript.txt`. The image command reuses the
existing bounded Pillow renderer and refuses overwrites. New runs use real
current timestamps and fresh UUIDs, so their artifacts need not be byte-identical
to the committed illustration; within one run the before/after exports must be.

## Portfolio walkthrough and limitations

Show the visible-page/complete-history distinction, all three separate
decisions, source provenance, and exact second-quote offsets. Open Markdown
and JSON with the server stopped; show the demo's real byte counts/digests,
unchanged events and zero forbidden-call measurements. Explain why snapshot
consistency, complete fail-not-truncate output and literal text are useful
engineering properties **without claiming scientific accuracy**.

This remains local and unauthenticated. Original questions, generated answers,
full passages, source metadata and human notes can be sensitive. URLs/logs and
saved files can retain IDs despite no-store/no-referrer. Deleting current
documents does not erase frozen artifacts, opinions or backups. Only share
reviewed material you have permission to redistribute; the demo is synthetic.

**Design inspiration, checked 2026-10-21 America/Los_Angeles:** the
[Zotero PDF reader documentation](https://www.zotero.org/support/pdf_reader#adding_annotations_to_notes)
describes creating notes from multiple annotations with citations and
source-context navigation, and [Zotero notes](https://www.zotero.org/support/notes)
describes saved child/standalone notes.
[Zotero](https://github.com/zotero/zotero) had about 15,692 stars at that check.
The local gap was different: yesterday's native notebook could browse opinions,
but no complete portable artifact combined them with frozen model evidence.
This implementation adds no Zotero synchronization/integration, PDF editing
or reference-manager format compatibility.

[RAGFlow](https://github.com/infiniflow/ragflow) (about 92,607 stars) and
[PaperQA](https://github.com/Future-House/paper-qa) (about 9,444 stars) provide
citation-oriented research context, not feature-parity claims or copied code.
Stars are dated context, not correctness evidence.

The stack is FastAPI, Pydantic, HTTPX, SQLite and a custom
Observe -> Decide -> Act state machine, **not LangGraph**. Default retrieval
uses deterministic hash embeddings/BM25, not learned semantic embeddings.
The [2026-10-21 catalog-only check](PROVIDER_MODELS_GUIDE.md) preserves
`gpt-6-astra`, `claude-sonnet-5-5`, `gemini-3.8-flash` (SPEED) and `kimi-k3`
defaults and separately dated migration contracts. Export uses none of them;
catalog checking is not a live inference or entitlement test.
