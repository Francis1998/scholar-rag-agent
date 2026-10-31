"""Measure native collection editing on synthetic persisted evidence; no model calls."""

import argparse
import hashlib
import json
import socket
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlencode, urlsplit

import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_corpus_explorer import ExplorerPage, _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_frozen_annotation import DOCUMENT_ID, RUN_ID
from scripts.demo_frozen_annotation import _seed as seed_frozen_evidence
from storage.paper_collections import PaperCollection, SQLitePaperCollections
from storage.paper_screening import ScreeningExport, ScreeningSubmission

FIXTURE = "scholar-native-collection-editing-synthetic-v1"
SOURCE = "synthetic:collection-editor"
PREFIX = "/library"
ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}
PANEL_TITLES = (
    "1. Open a bounded editor without writes",
    "2. Preview exact changes, then explicitly save",
    "3. Reject stale resubmissions without overwriting",
    "4. Restart with papers, evidence and opinions intact",
)


class EditingPage(ExplorerPage):
    """Inspect actual independent forms; unchecked choices are not successful controls."""

    def __init__(self, html: str) -> None:
        self.forms: dict[str, dict[str, str]] = {}
        self.actions: dict[str, str] = {}
        self.choices: dict[str, str] = {}
        self.preselected = 0
        self._active_form = ""
        super().__init__(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id"):
            self._active_form = str(attributes["id"])
            self.forms[self._active_form] = {}
            self.actions[self._active_form] = str(attributes["action"])
        if tag == "input" and self._active_form and attributes.get("name"):
            name, value = str(attributes["name"]), str(attributes.get("value", ""))
            if attributes.get("type") == "checkbox":
                self.choices[name] = value
                self.preselected += int("checked" in attributes)
            else:
                self.forms[self._active_form][name] = value

    def handle_endtag(self, tag: str) -> None:
        super().handle_endtag(tag)
        if tag == "form":
            self._active_form = ""


def seed_demo(container: AppContainer) -> PaperCollection:
    seed_frozen_evidence(container)
    documents = [
        Document(
            document_id=identifier, title=title, source=SOURCE, text="Original synthetic note."
        )
        for identifier, title in (
            ("01-retained", "Synthetic retained member"),
            ("02-addition", "Synthetic explicit addition"),
            ("03-off-page", "Synthetic off-page member"),
        )
    ]
    container.document_store.add_documents(
        documents,
        [
            Chunk(**document.model_dump(), chunk_id=document.document_id + "-chunk")
            for document in documents
        ],
    )
    collection = container.paper_collections.create(
        name="Synthetic before edit", document_ids=[DOCUMENT_ID, "01-retained", "03-off-page"]
    )
    for identifier, revision, decision in (
        (DOCUMENT_ID, 0, "exclude"),
        ("01-retained", 0, "unsure"),
        ("01-retained", 1, "include"),
    ):
        container.paper_screening.submit(
            collection.collection_id,
            identifier,
            ScreeningSubmission.model_validate(
                {
                    "collection_revision": 1,
                    "expected_decision_revision": revision,
                    "decision": decision,
                    "reason": "Synthetic human opinion, not scientific validation.",
                }
            ),
        )
    return collection


def protected_rows(database: Path) -> dict[str, list[tuple[object, ...]]]:
    queries = {
        "documents": "SELECT * FROM documents ORDER BY document_id",
        "chunks": "SELECT * FROM chunks ORDER BY chunk_id",
        "events": "SELECT * FROM agent_events ORDER BY id",
        "decisions": "SELECT * FROM paper_screening_decisions ORDER BY collection_id, document_id",
        "history": (
            "SELECT * FROM paper_screening_history ORDER BY collection_id, document_id, revision"
        ),
    }
    with closing(sqlite3.connect(database)) as connection:
        return {name: connection.execute(query).fetchall() for name, query in queries.items()}


@contextmanager
def editing_guards(container: AppContainer) -> Iterator[dict[str, Mock]]:
    with browsing_guards(container) as guards, ExitStack() as stack:
        for target, method in (
            (container.paper_screening, "submit"),
            (container.worksheets, "build"),
            (container.event_log, "append_transition"),
            (container.answer_reviews, "create"),
            (container.evidence_annotations, "create"),
            (socket, "create_connection"),
            (socket.socket, "connect"),
        ):
            guards[f"{getattr(target, '__name__', type(target).__name__)}.{method}"] = (
                stack.enter_context(patch.object(target, method, side_effect=_no_work))
            )
        yield guards


def run_demo(output_dir: Path) -> str:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    database = output_dir / "corpus.sqlite3"
    settings = offline_settings(database)
    application = create_app(settings)
    container: AppContainer = application.state.container
    collection = seed_demo(container)
    protected = protected_rows(database)
    frozen = container.saved_run_evidence.export(RUN_ID).model_dump_json()
    before_reads = database.read_bytes()
    artifacts: dict[str, bytes] = {}
    pages: dict[str, str] = {}
    with TestClient(application, root_path=PREFIX) as client, editing_guards(container) as guards:

        def get(name: str, url: str) -> EditingPage:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path.startswith(PREFIX + "/"):
                raise RuntimeError("Demo navigation left the local mounted app.")
            response = client.get(url)
            response.raise_for_status()
            if response.headers["cache-control"] != "no-store":
                raise RuntimeError("Response lost no-store.")
            artifacts[name + ".html"] = response.content
            pages[name] = url
            return EditingPage(response.text)

        listed = get("collections-before", PREFIX + "/explore/collections")
        first_url = (
            listed.links["edit-collection-1"] + "&" + urlencode({"limit": 1, "source": SOURCE})
        )
        first = get("editor-first", first_url)
        second = get("editor-next", first.links["next-page"])
        get_unchanged = database.read_bytes() == before_reads
        if (
            first.preselected
            or second.preselected
            or [
                json.loads(value)
                for key, value in second.choices.items()
                if key.startswith("remove_")
            ]
            != list(collection.document_ids)
            or second.choices.get("add_1") != '"02-addition"'
            or not get_unchanged
        ):
            raise RuntimeError("GET selections, exact IDs, paging or read-only boundary failed.")
        payload = {
            **second.forms["collection-edit"],
            "name": "Synthetic after edit",
            "remove_1": second.choices["remove_1"],
            "add_1": second.choices["add_1"],
        }
        preview = client.post(second.actions["collection-edit"], data=payload, headers=ORIGIN)
        preview.raise_for_status()
        artifacts["preview.html"] = preview.content
        reviewed = EditingPage(preview.text)
        preview_unchanged = database.read_bytes() == before_reads
        if not preview_unchanged or reviewed.forms["collection-save"] != {
            **payload,
            "action": "save",
        }:
            raise RuntimeError("Preview changed storage or changed the explicitly chosen delta.")
        saved = client.post(
            reviewed.actions["collection-save"],
            data=reviewed.forms["collection-save"],
            headers=ORIGIN,
            follow_redirects=False,
        )
        if saved.status_code != 303:
            raise RuntimeError("Native save did not return 303.")
        location = saved.headers["location"]
        get("collection-after", location)
        current = container.paper_collections.get(collection.collection_id)
        if (
            current.document_ids != ("01-retained", "03-off-page", "02-addition")
            or current.revision != 2
        ):
            raise RuntimeError("Explicit membership edit lost retained members or its revision.")
        after = database.read_bytes()
        statuses = []
        for index in range(2):
            stale = client.post(
                reviewed.actions["collection-save"],
                data=reviewed.forms["collection-save"],
                headers=ORIGIN,
                follow_redirects=False,
            )
            statuses.append(stale.status_code)
            artifacts[f"stale-{index + 1}.html"] = stale.content
            if "location" in stale.headers or database.read_bytes() != after:
                raise RuntimeError("A stale save overwrote data or reported success.")
        api_response = client.get(PREFIX + "/collections/" + collection.collection_id)
        api_response.raise_for_status()
        artifacts["collection-before.json"] = collection.model_dump_json(indent=2).encode()
        artifacts["collection-after.json"] = api_response.content
        api_equal = PaperCollection.model_validate_json(api_response.content) == current
        snapshot = ScreeningExport.model_validate_json(
            container.paper_screening.export_results(
                collection.collection_id, collection_revision=2
            ).content
        )
        counts = {name: guard.call_count for name, guard in guards.items()}
    restarted = create_app(settings)
    with (
        TestClient(restarted, root_path=PREFIX) as client,
        editing_guards(restarted.state.container) as guards,
    ):
        reopened = client.get(location)
        reopened.raise_for_status()
        artifacts["restarted.html"] = reopened.content
        counts.update({f"restart:{name}": guard.call_count for name, guard in guards.items()})
    checks = {
        "fixture": FIXTURE,
        "before_document_ids": list(collection.document_ids),
        "after_document_ids": list(current.document_ids),
        "before_name": collection.name,
        "after_name": current.name,
        "before_revision": collection.revision,
        "after_revision": current.revision,
        "retained_members": len(set(collection.document_ids) & set(current.document_ids)),
        "preselected_checkboxes": first.preselected + second.preselected,
        "preview_status": preview.status_code,
        "save_status": saved.status_code,
        "stale_save_statuses": statuses,
        "get_and_preview_database_unchanged": get_unchanged and preview_unchanged,
        "rejected_writes_database_unchanged": database.read_bytes() == after,
        "python_json_equal": api_equal,
        "restart_collection_equal": SQLitePaperCollections(database).get(collection.collection_id)
        == current,
        "restart_html_identical": artifacts["collection-after.html"] == artifacts["restarted.html"],
        "protected_table_rows_unchanged": protected_rows(database) == protected,
        "protected_row_counts": {name: len(rows) for name, rows in protected.items()},
        "frozen_evidence_unchanged": restarted.state.container.saved_run_evidence.export(
            RUN_ID
        ).model_dump_json()
        == frozen,
        "frozen_evidence_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
        "screening_counts_after": snapshot.counts.model_dump(),
        "forbidden_call_counts": counts,
    }
    boolean_checks = (
        "get_and_preview_database_unchanged",
        "rejected_writes_database_unchanged",
        "python_json_equal",
        "restart_collection_equal",
        "restart_html_identical",
        "protected_table_rows_unchanged",
        "frozen_evidence_unchanged",
    )
    if (
        statuses != [409, 409]
        or any(counts.values())
        or not all(checks[key] for key in boolean_checks)
        or snapshot.counts.stale != 1
        or snapshot.counts.unscreened != 2
        or snapshot.included_document_ids
    ):
        raise RuntimeError("Measured preservation, restart, staleness or no-call checks failed.")
    transcript = (
        f"{PANEL_TITLES[0]}\n"
        f"Saved members: {len(collection.document_ids)}; candidate page size: 1.\n"
        f"Preselected checkboxes: {checks['preselected_checkboxes']}; GET bytes unchanged.\n"
        "Next page changes candidates only; all removals still show exact JSON IDs.\n\n"
        f"{PANEL_TITLES[1]}\n"
        f"Preview: {preview.status_code}; no writes. Remove 1, add 1, retain 2.\n"
        f"Save: {saved.status_code}; revision {collection.revision} -> {current.revision}; "
        f"members {len(collection.document_ids)} -> {len(current.document_ids)}.\n"
        "Renamed Synthetic before edit -> Synthetic after edit.\n"
        "The off-page member is retained; the removed paper is not deleted.\n\n"
        f"{PANEL_TITLES[2]}\n"
        f"Repeated old-revision saves: {statuses}; database unchanged.\n"
        f"Current screening: stale={snapshot.counts.stale}, "
        f"unscreened={snapshot.counts.unscreened}.\n"
        "Earlier include is stale, not current; opinions/history remain stored.\n\n"
        f"{PANEL_TITLES[3]}\n"
        "Python/JSON and restarted collection equal; restarted HTML identical.\n"
        f"Preserved rows: papers={len(protected['documents'])}, chunks={len(protected['chunks'])}, "
        f"events={len(protected['events'])}, decisions={len(protected['decisions'])}, "
        f"history={len(protected['history'])}.\n"
        "Frozen evidence unchanged; forbidden model/retrieval/network/write calls: "
        f"{sum(counts.values())}.\n"
        "Executed synthetic HTTP measurements, not browser footage or scientific findings.\n"
    )
    for name, value in (("checks.json", checks), ("pages.json", pages)):
        artifacts[name] = (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode()
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)
    return transcript


def serve_demo(output_dir: Path, port: int) -> None:
    database = output_dir / "corpus.sqlite3"
    checks = json.loads((output_dir / "checks.json").read_text(encoding="utf-8"))
    if checks.get("fixture") != FIXTURE or not database.is_file():
        raise ValueError("Expected an existing native collection editing demo directory.")
    app = create_app(offline_settings(database))
    mounted = FastAPI()
    mounted.mount(PREFIX, app)
    with editing_guards(app.state.container):
        uvicorn.run(mounted, host="127.0.0.1", port=port, access_log=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", type=Path)
    action.add_argument("--serve", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    if arguments.serve:
        serve_demo(arguments.serve, arguments.port)
    else:
        print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
