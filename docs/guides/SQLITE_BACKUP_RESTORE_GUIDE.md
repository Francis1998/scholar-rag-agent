# Back up and recover a local SQLite research workspace

Preserve the database, then prove that a **newly restored copy** contains the
same papers, selections, human decisions and frozen evidence. This walkthrough
uses Python 3.11+'s standard-library `sqlite3.Connection.backup`, the existing
app factory and synthetic demo fixtures. There is no new backup command,
restore endpoint, dependency or automatic recovery service.

## What to preserve

| Artifact | Preserves | Does not replace |
| --- | --- | --- |
| A verified SQLite backup | All committed tables in that database: documents/chunks and metadata, stored graph records, collections/revisions, latest screening decisions and retained history, raw run events and captured evidence, answer reviews and exact annotations; any optional tables stored in the same file | Original publication files, configuration, credentials, application code, or other databases |
| [Portable notebook JSON/Markdown](PORTABLE_REVIEW_NOTEBOOK_GUIDE.md) | One completed run's frozen evidence and both complete retained opinion histories, within export limits | A whole-corpus/database backup or a supported database import |
| [Evidence JSON/Markdown](EVIDENCE_EXPORT_GUIDE.md), HTML or bibliography | The selected run's supported captured evidence or cited references, according to that format's contract | All raw events, unrelated runs, collections, screening history or an entire publication |
| Separately retained permitted PDF/text/source copies | The actual source files, including content never ingested or captured | Database identities, review history or application state |

Keep the code revision, dependency lock, Python/SQLite versions and non-secret
settings with your backup inventory. Store secrets separately using your own
protected credential workflow; do not put them in a manifest or commit a local
environment file. Database backups can nevertheless contain secrets or personal
information previously pasted into text, questions, notes or diagnostic events.
Protect backups, exports and logs as sensitively as the original corpus.

## Maintenance boundary: stop writers, keep the original

1. Identify the **actual absolute database path** used by the running instance.
   `SCHOLAR_RAG_DATABASE_PATH` can override the default
   `.scholar-rag-agent.sqlite3`, and a relative path depends on the process's
   working directory. Do not start another app just to discover the path:
   startup can create an empty database or initialize schema.
2. Stop new submissions and let active ingestion, queries, collection edits,
   screening, reviews and annotations finish. Gracefully stop all API workers,
   reloaders and other scripts/jobs using that file. Keep them stopped through
   backup and verification. Close database-browser write sessions too.
3. Use trusted local directories with sufficient free space. Never point a
   restore at the running/original database, overwrite an existing destination,
   or try to swap a database underneath a process. The snippets reject existing
   destination files and symlinks; **they do not discover or stop services**.
   A fresh pathname is not a lock against another process starting there.
4. Preserve an untouched verified backup. Restore into another new path, using
   the same code revision and locked dependencies first. Startup initializes
   tables/indexes; it is not a read-only schema inspection or a promised
   cross-version migration. Consult the [migration notes](../../migrations/README.md)
   before changing application versions.

Do **not** copy just a live database file with a file manager, `cp` or `rsync`.
In WAL mode, committed data can still live in its `-wal` sidecar; copying a
database and its sidecars separately is not an atomic snapshot either. Rollback
journals also participate in crash recovery. Do not delete, rename, checkpoint
or change the source's journal mode to make a copy appear usable.

The backup API reads through SQLite, including committed WAL content, and
produces a consistent database image. It can accommodate concurrent writers,
but that is **not a whole-application transactional snapshot**: one query writes
several events over time; an in-flight workflow may be only partly committed.
Uncommitted work, in-memory state and transactions in other databases are not
included. This guide deliberately requires a maintenance window.

Read-only SQLite access can still need existing WAL/shared-memory files or
permission to create sidecars. Do not add `immutable=1` to bypass locking/WAL
handling. If a crashed source needs journal recovery, cannot be opened read-only,
or fails checks, stop: preserve the original database and sidecars together for
specialist recovery. These commands are not corruption repair or deleted-data
recovery tools.

## 1. Prepare an isolated practice workspace

Commands assume a POSIX shell at the repository root with the existing
[development environment](../../QUICKSTART.md) installed. If dependencies are
missing, restore them with `uv sync --frozen --extra dev`.

The default exercise uses **only new synthetic files**. It does not inspect a
personal database, environment file or provider credentials:

```bash
umask 077
export RECOVERY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/scholar-recovery.XXXXXX")"
export RECOVERY_DIR="$(cd "$RECOVERY_DIR" && pwd -P)"
export ORIGINAL_DB="$RECOVERY_DIR/original.sqlite3"
export BACKUP_DB="$RECOVERY_DIR/backup.sqlite3"
export RESTORED_DB="$RECOVERY_DIR/restored.sqlite3"
printf 'Recovery directory: %s\n' "$RECOVERY_DIR"
git rev-parse HEAD
```

For a real maintenance operation, instead set `ORIGINAL_DB` to the verified,
stopped database and choose **distinct, unused absolute** `BACKUP_DB` and
`RESTORED_DB` paths in private, durable directories. Their parent directories
must already exist. Skip the synthetic setup and fixture-specific assertions;
use your own recorded IDs and expected results. Temporary directories are for
practice, **not your only durable backup**. A backup on the same disk does not
protect against losing that disk.

## 2. Create a small synthetic research record

This reuses the existing notebook demo's `_seed` fixture, not its destructive
corpus-change demonstration. It creates one fake-adapter run, three independent
answer reviews and two annotations on different occurrences of the same quote.
It then saves a collection and two screening decisions. The demo helper
`offline_settings` accepts only explicit settings, ignores environment/secret
sources and leaves all four provider keys blank. Socket guards reject network
connections. These demo helpers are tied to the checked-out revision, not a
new public backup API.

```bash
uv run --no-sync python - <<'PY'
import hashlib
import json
import os
import socket
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from fastapi.testclient import TestClient

from api.application import create_app
from scripts.demo_evidence_export import offline_settings
from scripts.demo_review_notebook import _seed

root = Path(os.environ["RECOVERY_DIR"])
database = Path(os.environ["ORIGINAL_DB"])
if not root.is_absolute() or database != root / "original.sqlite3":
    raise SystemExit("The fixture may only use its new practice directory.")
with database.open("xb"):
    pass
expected = {}
with (
    patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")),
    patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden")),
):
    app = create_app(offline_settings(database))
    with TestClient(app) as client:
        bundle = _seed(client, app.state.container)
        source = bundle.snapshot.sources[0]
        document_id = source.chunk.document_id
        response = client.post(
            "/collections",
            json={"name": "Synthetic recovery selection", "document_ids": [document_id]},
        )
        response.raise_for_status()
        collection = response.json()
        base = f"/collections/{collection['collection_id']}"
        revision = collection["revision"]
        for previous, decision in enumerate(("unsure", "include")):
            response = client.put(
                base + "/screening/" + quote(document_id, safe=""),
                json={
                    "collection_revision": revision,
                    "expected_decision_revision": previous,
                    "decision": decision,
                    "reason": f"Synthetic {decision} opinion, not scientific validation.",
                },
            )
            response.raise_for_status()
        requests = [
            ("/documents", {"limit": 100}),
            ("/runs", {"limit": 100}),
            (base, {}),
            (base + "/screening", {"collection_revision": revision, "limit": 100}),
            (base + "/screening/history", {
                "document_id": document_id, "collection_revision": revision, "limit": 100,
            }),
            ("/explore/run/notebook/export", {"run_id": bundle.run_id, "format": "json"}),
            ("/explore/run/notebook/export", {"run_id": bundle.run_id, "format": "markdown"}),
        ]
        expected["responses"] = []
        for url, params in requests:
            response = client.get(url, params=params)
            response.raise_for_status()
            expected["responses"].append([url, params, response.text])
        expected.update(
            run_id=bundle.run_id, document_id=document_id, chunk_id=source.chunk.chunk_id,
            collection_id=collection["collection_id"],
            events=app.state.container.event_log.list_events(),
        )
with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
    assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    expected["sql_sha256"] = hashlib.sha256(
        "\n".join(connection.iterdump()).encode("utf-8")
    ).hexdigest()
expected["original_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
with (root / "expected.json").open("x", encoding="utf-8") as output:
    json.dump(expected, output, ensure_ascii=False, indent=2)
print("Saved synthetic expectations:", root / "expected.json")
print("Original run:", expected["run_id"])
print("Collection:", expected["collection_id"])
print("Provider: fake; reviews: 3; annotations: 2; screening revisions: 2")
PY
```

The process exits before backup, leaving no fixture app running. The small
`expected.json` is a private exercise artifact containing original responses,
IDs, events and fingerprints, not an export/import format. Its SQL fingerprint
covers every table, including graph records and sequence counters. Whole-file
reads and `iterdump` here are suitable for this tiny fixture, not a suggested
large-corpus validation strategy.

## 3. Back up into a new file

Select the source and fresh destination:

```bash
export SOURCE_DB="$ORIGINAL_DB" DEST_DB="$BACKUP_DB"
```

Run this **copy block**. It never uses the app factory on the source, never
creates a missing source and never opens an existing destination for writing.
Directory aliases are resolved; final-component symlinks, including dangling
destination links, are rejected. Exclusive creation protects an existing
pathname, not against hostile directory replacement: use a private, trusted
directory and keep competing processes stopped.

```bash
uv run --no-sync python - <<'PY'
import hashlib
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

source = Path(os.environ["SOURCE_DB"])
target = Path(os.environ["DEST_DB"])
if not source.is_absolute() or not target.is_absolute():
    raise SystemExit("Use explicit absolute source and destination paths.")
if source.is_symlink() or target.is_symlink():
    raise SystemExit("Refusing a source or destination symlink.")
source = source.parent.resolve(strict=True) / source.name
target = target.parent.resolve(strict=True) / target.name
if not source.is_file() or source.stat().st_size == 0:
    raise SystemExit("Source must be an existing, nonempty regular database file.")
if source == target:
    raise SystemExit("Source and destination must be distinct.")
if any(os.path.lexists(str(target) + suffix) for suffix in ("", "-wal", "-shm", "-journal")):
    raise SystemExit("Destination or a destination sidecar already exists; choose a fresh path.")

deadline = time.monotonic() + 60
def progress(status, remaining, total):
    if time.monotonic() >= deadline:
        raise TimeoutError("Backup exceeded its cooperative 60-second budget.")

with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=5)) as reader:
    reader.execute("SELECT name FROM sqlite_schema LIMIT 1").fetchone()
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    with closing(sqlite3.connect(target.as_uri() + "?mode=rw", uri=True, timeout=5)) as writer:
        reader.backup(writer, pages=256, progress=progress, sleep=0.1)
with closing(sqlite3.connect(target.as_uri() + "?mode=ro", uri=True, timeout=5)) as check:
    if check.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise RuntimeError("Backup integrity_check failed; do not use this destination.")
with target.open("rb") as saved:
    digest = hashlib.file_digest(saved, "sha256").hexdigest()
print(f"Verified SQLite image: {target}")
print(f"integrity_check=ok; SQLite={sqlite3.sqlite_version}; sha256={digest}")
PY
```

Only the final success lines mark a verified image. A failed/interrupted command
can leave a partial destination and sidecars; do not use them, overwrite them
on retry, or remove anything from the source. Inspect the error and choose a new
destination. The five-second connection timeout is a busy-lock timeout, not a
whole-operation deadline. The progress budget is cooperative, and does not
bound a blocked filesystem call or the subsequent full integrity scan.

`integrity_check=ok` establishes SQLite structural consistency, **not** valid
application records, complete scientific evidence or authenticated provenance.
A file digest detects changes against a trusted inventory; it is not a
signature. For durable retention, close the connections first, retain the
verified image and its digest on protected storage, keep multiple generations,
and rehearse recovery from a separate protected copy.

## 4. Restore the backup into another new path

Do not re-ingest publications or POST a new query to reconstruct old state.
Those operations cannot restore original run IDs, human timestamps or frozen
annotations.

```bash
export SOURCE_DB="$BACKUP_DB" DEST_DB="$RESTORED_DB"
```

**Run the exact copy block from section 3 again**, unchanged. It now reads the
backup and verifies the new restored image. Leave both the original and backup
in place; do not rename the restored file over either of them.

For the synthetic exercise, re-run the copy block with the following inputs to
check refusal behavior. Each must exit nonzero before creating/modifying the
destination. Restore the section-4 variables afterwards.

| `SOURCE_DB` | `DEST_DB` | Expected refusal |
| --- | --- | --- |
| `$RECOVERY_DIR/missing.sqlite3` (does not exist) | `$RECOVERY_DIR/must-stay-absent.sqlite3` | Missing source; neither file is created |
| `$BACKUP_DB` | `$ORIGINAL_DB` | Existing destination; original stays unchanged |
| `$BACKUP_DB` | `$BACKUP_DB` | Identical source/destination |
| `$BACKUP_DB` | An existing or dangling symlink in the practice directory | Symlink; link and referent stay unchanged |
| `$BACKUP_DB` | `$RESTORED_DB` while its isolated server is running | Existing destination; no active database is replaced |

## 5. Prove the synthetic recovery without regenerating

This starts the restored app **in process**, with the same fake-only settings.
It compares original catalog/run IDs, collection membership/revisions, latest
screening decisions, full retained screening history and both complete notebook
formats. Exact response equality includes original review/annotation UUIDs,
sequences, reasons/comments/notes, timestamps, offsets, quotes and digests.
It also verifies every persisted table and raw event, not just exported fields.

```bash
uv run --no-sync python - <<'PY'
import hashlib
import json
import os
import socket
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.application import create_app
from scripts.demo_evidence_export import offline_settings
from storage.review_notebook import ReviewNotebook

root = Path(os.environ["RECOVERY_DIR"])
original, backup, restored = (
    Path(os.environ[name]) for name in ("ORIGINAL_DB", "BACKUP_DB", "RESTORED_DB")
)
if (original, backup, restored) != tuple(
    root / name for name in ("original.sqlite3", "backup.sqlite3", "restored.sqlite3")
):
    raise SystemExit("These assertions are only for the isolated synthetic exercise.")
expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def check_rows(path):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Expected an existing regular image, not a symlink.")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        digest = hashlib.sha256("\n".join(connection.iterdump()).encode("utf-8")).hexdigest()
        assert digest == expected["sql_sha256"], f"Persisted rows changed: {path}"

for image in (original, backup, restored):
    check_rows(image)
assert file_hash(original) == expected["original_sha256"]
backup_hash = file_hash(backup)
with (
    patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")),
    patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden")),
):
    app = create_app(offline_settings(restored))
    with (
        TestClient(app) as client,
        patch.object(app.state.container.llm, "generate",
                     side_effect=AssertionError("Recovery must not generate")),
    ):
        for url, params, body in expected["responses"]:
            response = client.get(url, params=params)
            response.raise_for_status()
            assert response.text == body, f"Recovered response differs: {url}"
        response = client.get(
            "/explore/run/notebook/export",
            params={"run_id": expected["run_id"], "format": "json"},
        )
        response.raise_for_status()
        notebook = ReviewNotebook.model_validate_json(response.content)
        assert len(notebook.reviews) == 3 and len(notebook.annotations) == 2
        assert len({item.start for item in notebook.annotations}) == 2
        for annotation in notebook.annotations:
            source = next(item for item in notebook.evidence.snapshot.sources
                          if item.chunk.chunk_id == annotation.chunk_id)
            text = source.chunk.text
            assert source.chunk.document_id == annotation.document_id
            assert text[annotation.start:annotation.end] == annotation.quote
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == annotation.source_text_sha256
        preview = client.post("/retrieve", json={
            "query": notebook.evidence.query, "document_ids": [expected["document_id"]],
        })
        preview.raise_for_status()
        assert {item["chunk"]["chunk_id"] for item in preview.json()["sources"]} == {
            expected["chunk_id"]
        }
        assert app.state.container.event_log.list_events() == expected["events"]
for image in (original, backup, restored):
    check_rows(image)
assert file_hash(original) == expected["original_sha256"]
assert file_hash(backup) == backup_hash
print("PASS: all three integrity checks and complete persisted rows match")
print("PASS: original IDs/events/collections/screening history/notebooks recovered")
print("PASS: 3 reviews, 2 exact quote occurrences, rebuilt retrieval indexes")
print("PASS: original bytes and backup unchanged; no network or recovery generation")
PY
```

The normal app builds BM25 and deterministic hash-dense indexes from stored
chunks at startup and uses persisted graph records. A successful scoped
`/retrieve` checks that the new instance can actually retrieve restored chunks,
not merely display an old answer. Restart **every** app/container after choosing
a restored database; an older process retains its own indexes and must not be
repointed by replacing its open database file.

## 6. Open only the restored copy on loopback

Use an unused port; if binding fails, do not stop an unrelated service. This
explicit factory invocation avoids importing `api.main`, which initializes
the default app on import. The existing offline settings helper ignores all
ambient settings, including a different database path or provider selection.

```bash
OPENAI_API_KEY= ANTHROPIC_API_KEY= GEMINI_API_KEY= MOONSHOT_API_KEY= \
  uv run --no-sync python - <<'PY'
import os
from pathlib import Path
import uvicorn

from api.application import create_app
from scripts.demo_evidence_export import offline_settings

database = Path(os.environ["RESTORED_DB"])
if not database.is_absolute() or database.is_symlink() or not database.is_file():
    raise SystemExit("Refusing to start without an existing restored image.")
if database.resolve() in {
    Path(os.environ["ORIGINAL_DB"]).resolve(), Path(os.environ["BACKUP_DB"]).resolve()
}:
    raise SystemExit("Do not start this inspection server on the original or backup.")
uvicorn.run(create_app(offline_settings(database)), host="127.0.0.1", port=8017)
PY
```

In another terminal:

```bash
curl --fail-with-body --silent --show-error http://127.0.0.1:8017/health
curl --fail-with-body --silent --show-error 'http://127.0.0.1:8017/runs?limit=20'
```

Open `http://127.0.0.1:8017/explore/runs`, choose the **original** run ID, inspect
its human-review notebook and follow both exact frozen quote occurrences.
Browse saved collections and the member's screening history. Do not generate
another answer as a substitute for these checks. Stop this server with Ctrl-C.
For a real recovery, only after acceptance explicitly configure the intended
service to use the verified new path and restart it with reviewed settings.
Keep the old database and protected backup; this guide performs no replacement,
cleanup, scheduled backup or automatic cutover.

## Recovery limits and an honest portfolio demonstration

Record the actual revision, original/restored IDs, integrity results, unchanged
original fingerprint, preserved opinion counts/selectors and successful
generation-free preview. Show the restarted local notebook and screening
history, explaining that these are synthetic software checks, **not research
findings or authenticated reviewer judgments**. There is no need to download
publications, use a paid model or produce another animation.

Recovery is model-independent. This stack is Python/FastAPI/Pydantic, HTTPX,
SQLite, BM25/hash-dense/graph retrieval and a hand-written agent state machine.
The [provider model guide](PROVIDER_MODELS_GUIDE.md) covers normal live defaults;
this walkthrough neither calls those providers nor validates their availability.

For real data, compare trusted recorded IDs and representative stored records
before resuming work. Follow pagination: catalog/run/review/annotation/history
pages are bounded and are not whole-database exports. A portable notebook is
one completed captured run, capped at 1,000 reviews, 1,000 annotations and 8 MiB
per representation, with additional event/history byte bounds. An HTTP 413 is
not an empty or successful backup. Screening exports cover current selections,
not every historical opinion. See the linked format guides for exact limits.
SQLite backup is independent of these endpoint caps but still needs disk space,
SQLite access, compatible application records and a meaningful acceptance test.

A backup cannot recover commits made after it, overwritten legacy screening
reasons never retained, missing legacy snapshots, source files never stored, or
uncommitted work lost in a crash. Restoring events does not resume a nonterminal
run or replay an external model deterministically. SQLite integrity checks do
not validate all application references, annotation anchors or scientific
claims; the restored readers can still explicitly reject corrupt records.

This remains a local unauthenticated service. Blank keys and fake inference are
not an access-control boundary. Backups preserve **all** stored data, including
raw events excluded from privacy-filtered exports and opinions retained after
current-corpus deletion; they are not redacted or necessarily free of residual
deleted content. No-store headers do not protect downloaded artifacts.
Use appropriate permissions, encrypted/off-device storage and retention rules;
do not upload a personal database, expectation file or notebook to a public PR.
