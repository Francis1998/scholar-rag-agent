"""Execute an offline reference import: RIS + CSL-JSON -> preview -> commit -> collection view.

Writes only a new temporary database and the requested output directory. Both input files
are synthetic and saved next to the outputs. The real browser routes preview and commit the
import; the JSON API then shows that re-importing is idempotent. Network, models, retrieval
and agent events are guarded and must never be called.
"""

import argparse
import hashlib
import json
import re
import socket
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from scripts.demo_evidence_export import offline_settings

ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}
COLLECTION_NAME = "Sleep and caffeine review"
PANEL_TITLES = (
    "1. Two exports of one search, with an overlapping paper",
    "2. Preview (dry run): everything reported, nothing written",
    "3. Commit: one explicit browser POST, then the collection view",
    "4. Re-import is idempotent; labels never decide",
)
OUTPUT_NAMES = (
    "pubmed-search.ris",
    "zotero-library.json",
    "form.html",
    "preview.html",
    "collection.html",
    "reimport-preview.json",
    "history.json",
    "checks.json",
    "transcript.txt",
)
RIS = """TY  - JOUR
TI  - Caffeine timing and sleep onset in adolescents: a randomised crossover trial
AU  - Garcia, María José
AU  - Østergaard, Søren
PY  - 2021
DO  - https://doi.org/10.5555/SLEEP.2021.001
AB  - Evening caffeine delayed sleep onset in a synthetic crossover cohort.
N1  - ASReview_relevant
ER  - 

TY  - JOUR
AB  - An exported record that lost its title.
ER  - 

TY  - JOUR
TI  - Blue light exposure and melatonin suppression in shift workers
AU  - Chen, Wei
PY  - 2019
DO  - 10.5555/LIGHT.2019.042
AB  - A synthetic abstract about evening light and melatonin.
ER  - 
"""
CSL = [
    {
        "id": "zotero-1",
        "type": "article-journal",
        "title": "Caffeine Timing and Sleep Onset in Adolescents: A Randomised Crossover Trial",
        "DOI": "10.5555/sleep.2021.001",
        "author": [{"family": "Garcia", "given": "María José"}],
        "issued": {"date-parts": [[2021, 3]]},
    },
    {
        "id": "zotero-2",
        "type": "article-journal",
        "title": "Napping and declarative memory consolidation in older adults",
        "author": [{"family": "Okafor", "given": "Ada"}],
        "issued": {"date-parts": [[2020]]},
        "PMID": "44556677",
        "abstract": "A synthetic abstract about naps and memory.",
    },
]


def _no_work(*args: object, **kwargs: object) -> None:
    raise AssertionError("The import demo attempted network, model, retrieval or event work.")


def _count(page: str, identifier: str) -> int:
    match = re.search(rf'id="{identifier}">([0-9]+)<', page)
    if match is None:
        raise ValueError(f"Missing {identifier} in the preview page.")
    return int(match.group(1))


def _details(page: str) -> tuple[list[object], list[object]]:
    """Read the first skipped record and in-import duplicate exactly as the page shows them."""
    skipped = re.search(
        r'File ([0-9]+), record ([0-9]+), line ([0-9]+): <span class="skip-reason">([a-z_]+)<',
        page,
    )
    duplicate = re.search(
        r'data-status="duplicate_in_import">.*?<dd>file ([0-9]+), record ([0-9]+)[^<]*</dd>'
        r'.*?class="match-reason">([a-z_]+)</span>[^<]*\(file ([0-9]+), record ([0-9]+)\)',
        page,
        re.DOTALL,
    )
    if skipped is None or duplicate is None:
        raise ValueError("The preview page does not list the skipped record and duplicate.")
    return (
        [int(skipped.group(n)) for n in (1, 2, 3)] + [skipped.group(4)],
        [int(duplicate.group(n)) for n in (1, 2, 4, 5)] + [duplicate.group(3)],
    )


def _guards(stack: ExitStack, container: AppContainer) -> dict[str, Mock]:
    targets: tuple[tuple[object, str], ...] = (
        (socket, "create_connection"),
        (socket.socket, "connect"),
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (container.llm, "generate"),
        (container.runner, "run"),
        (container.runner, "preview"),
        (container.hybrid_retriever, "retrieve"),
        (container.event_log, "append_event"),
    )
    return {
        f"{getattr(target, '__name__', type(target).__name__)}.{method}": stack.enter_context(
            patch.object(target, method, side_effect=_no_work)
        )
        for target, method in targets
    }


def _documents(container: AppContainer) -> list[dict[str, Any]]:
    catalog = container.document_catalog.list_documents(limit=100)
    return [document.model_dump() for document in catalog.documents]


def run_demo(output_dir: Path) -> str:
    """Save real response bytes and measured checks; refuse to overwrite anything."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    artifacts: dict[str, bytes] = {}

    def save(name: str, content: bytes) -> None:
        with (output_dir / name).open("xb") as output:
            output.write(content)
        artifacts[name] = content

    ris = RIS.encode("utf-8")
    csl = (json.dumps(CSL, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    save("pubmed-search.ris", ris)
    save("zotero-library.json", csl)
    uploads = [
        ("files", ("pubmed-search.ris", ris, "application/x-research-info-systems")),
        ("files", ("zotero-library.json", csl, "application/json")),
    ]
    with TemporaryDirectory(prefix="scholar-reference-import-") as temporary, ExitStack() as stack:
        database = Path(temporary) / "synthetic.sqlite3"
        app = create_app(offline_settings(database))
        container: AppContainer = app.state.container
        guards = _guards(stack, container)
        client = stack.enter_context(TestClient(app))
        form = client.get("/explore/collections/import")
        save("form.html", form.content)
        preview = client.post(
            "/explore/collections/import",
            files=uploads,
            data={"new_collection_name": COLLECTION_NAME},
            headers=ORIGIN,
        )
        save("preview.html", preview.content)
        documents_before = len(_documents(container))
        collections_before = len(container.paper_collections.list_collections().collections)
        draft = re.search(r'name="draft_id" value="(imd_[0-9a-f]{32})"', preview.text)
        if form.status_code != 200 or preview.status_code != 200 or draft is None:
            raise RuntimeError("The browser preview did not render a commit form.")
        committed = client.post(
            "/explore/collections/import/commit",
            data={"draft_id": draft.group(1)},
            headers=ORIGIN,
            follow_redirects=False,
        )
        location = committed.headers["location"]
        view = client.get(location)
        save("collection.html", view.content)
        (summary,) = container.paper_collections.list_collections().collections
        collection = container.paper_collections.get(summary.collection_id)
        history = client.get(f"/collections/{collection.collection_id}/imports")
        save("history.json", history.content)
        record = history.json()["imports"][0]
        documents = _documents(container)
        again = client.post(
            "/collections/imports/preview",
            files=uploads,
            data={
                "collection_id": collection.collection_id,
                "collection_revision": str(collection.revision),
            },
        )
        save("reimport-preview.json", again.content)
        repeat = client.post("/collections/imports", json={"draft_id": again.json()["draft_id"]})
        queue = container.paper_screening.list_queue(
            collection.collection_id, collection_revision=collection.revision, limit=100
        )
        unchanged = container.paper_collections.get(collection.collection_id) == collection
        with container.paper_collections.resolved_collection(collection.collection_id) as (
            connection,
            _,
        ):
            first_paper = json.loads(
                connection.execute(
                    "SELECT metadata FROM documents WHERE document_id = ?",
                    (record["papers"][0]["document_id"],),
                ).fetchone()[0]
            )
        calls = {name: guard.call_count for name, guard in guards.items()}
    reimport = again.json()
    reasons: dict[str, int] = {}
    for item in reimport["candidates"]:
        reasons[item["match_reason"]] = reasons.get(item["match_reason"], 0) + 1
    checks: dict[str, Any] = {
        "form_status": form.status_code,
        "preview": {
            name: _count(preview.text, name)
            for name in (
                "records-read",
                "records-skipped",
                "duplicates-in-import",
                "duplicates-in-collection",
                "papers-to-add",
            )
        },
        "skipped_detail": _details(preview.text)[0],
        "duplicate_detail": _details(preview.text)[1],
        "documents_before_commit": documents_before,
        "collections_before_commit": collections_before,
        "commit": [committed.status_code, location.split("?", 1)[0]],
        "members": len(collection.document_ids),
        "revision": collection.revision,
        "view_lists_members": all(
            identifier in view.text for identifier in collection.document_ids
        ),
        "files": [
            [
                file["file_name"],
                file["records_read"],
                file["papers_to_add"],
                file["duplicates_in_import"],
            ]
            for file in record["files"]
        ],
        "skipped": record["files"][0]["skipped_reasons"],
        "provenance": [
            first_paper["import_file"],
            first_paper["import_record_index"],
            first_paper.get("import_line"),
            first_paper["import_file_sha256"] == hashlib.sha256(ris).hexdigest(),
        ],
        "documents_after_commit": len(documents),
        "reimport": [reimport["totals"]["papers_to_add"], reimport["blocking_reason"]],
        "reimport_reasons": dict(sorted(reasons.items())),
        "previously_imported": [bool(file["previously_imported"]) for file in reimport["files"]],
        "repeat_commit": [repeat.status_code, repeat.json()["detail"]["code"]],
        "unchanged_after_repeat": unchanged,
        "labels": reimport["totals"]["labels"],
        "queue": [queue.counts.unscreened, queue.counts.include, queue.counts.exclude],
        "forbidden_call_counts": calls,
        "database_removed": not database.exists(),
    }
    checks["output_bytes"] = {name: len(content) for name, content in artifacts.items()}
    checks["sha256"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()
    }
    save("checks.json", (json.dumps(checks, indent=2, sort_keys=True) + "\n").encode())
    transcript = _transcript(checks)
    save("transcript.txt", transcript.encode("utf-8"))
    return transcript


def _transcript(checks: dict[str, Any]) -> str:
    preview, files = checks["preview"], checks["files"]
    (_, ris_records, ris_new, _), (_, csl_records, csl_new, csl_duplicates) = files
    calls = sum(checks["forbidden_call_counts"].values())
    reasons = ", ".join(f"{name} {count}" for name, count in checks["reimport_reasons"].items())
    labels = checks["labels"]
    unscreened, include, exclude = checks["queue"]
    file_name, record_index, line, sha_matches = checks["provenance"]
    panels = (
        (
            f"pubmed-search.ris   RIS        {ris_records} records (record 2 has no title)",
            f"zotero-library.json CSL-JSON   {csl_records} records",
            "Overlap: https://doi.org/10.5555/SLEEP.2021.001 (RIS) = 10.5555/sleep.2021.001",
            "RIS record 1 carries an ASReview_relevant label in its N1 note.",
            f"Offline: network, model, retrieval and event guards were called {calls} times.",
        ),
        (
            f"Records read {preview['records-read']} | skipped {preview['records-skipped']} | "
            f"duplicates within the import {preview['duplicates-in-import']} | "
            f"new papers {preview['papers-to-add']}",
            "Skipped: file {}, record {}, line {}: {}".format(*checks["skipped_detail"]),
            "Duplicate: file {}, record {} matches file {}, record {} by {}".format(
                *checks["duplicate_detail"]
            ),
            f"Labels in the files (ignored): {labels['included']} included, "
            f"{labels['excluded']} excluded",
            f"Before commit: {checks['documents_before_commit']} documents, "
            f"{checks['collections_before_commit']} collections (only a private 24-hour draft)",
        ),
        (
            f"POST /explore/collections/import/commit -> {checks['commit'][0]} "
            f"{checks['commit'][1]}",
            f"'{COLLECTION_NAME}': revision {checks['revision']}, {checks['members']} members",
            f"Import record: file 1 -> {ris_new} new; file 2 -> {csl_new} new, "
            f"{csl_duplicates} duplicate",
            f"Paper provenance: {file_name}, record {record_index}, line {line}, "
            f"file SHA-256 matches: {str(sha_matches).lower()}",
            f"Corpus documents: {checks['documents_before_commit']} before, "
            f"{checks['documents_after_commit']} after commit; collections: "
            f"{checks['collections_before_commit']} before, 1 after",
        ),
        (
            f"Same two files again: {checks['reimport'][0]} new; already in the collection "
            f"by {reasons}",
            "Both files are flagged as previously imported into this collection",
            f"Commit -> {checks['repeat_commit'][0]} {checks['repeat_commit'][1]}; "
            f"membership unchanged: {str(checks['unchanged_after_repeat']).lower()}",
            f"Screening queue: {unscreened} unscreened, {include} include, {exclude} exclude",
            "Fuzzy near-duplicates: run Inspect duplicate candidates on the collection.",
        ),
    )
    return (
        "\n\n".join(
            "\n".join((title, *lines)) for title, lines in zip(PANEL_TITLES, panels, strict=True)
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(run_demo(args.output_dir), end="")


if __name__ == "__main__":
    main()
