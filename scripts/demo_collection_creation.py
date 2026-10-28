"""Measure native collection creation using a new persistent, credential-free synthetic corpus."""

import argparse
import json
import socket
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlencode, urlsplit

import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_corpus_explorer import ExplorerPage, browsing_guards
from scripts.demo_evidence_export import offline_settings
from storage.paper_collections import PaperCollection, SQLitePaperCollections

FIXTURE = "scholar-native-collection-creation-synthetic-v1"
SOURCE = "synthetic:collection-builder"
SELECTED_IDS = ("01-methods/../paper\r\nid\0", "02-limitations/\u7814\u7a76")
NAME = "Synthetic native selection"
PREFIX = "/library"
PANEL_TITLES = (
    "1. Discover exact papers without writes",
    "2. Explicitly save one named selection",
    "3. Reject unsafe or invalid submissions atomically",
    "4. Reopen after restart and choose the next action",
)


class CreationPage(ExplorerPage):
    """Read only the explicit creation controls, not the independent GET filter form."""

    def __init__(self, html: str) -> None:
        self.form_action = ""
        self.name = ""
        self.choices: dict[str, str] = {}
        self.preselected = 0
        self._creation = False
        super().__init__(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id") == "collection-create":
            self._creation = True
            self.form_action = str(attributes["action"])
        if self._creation and tag == "input":
            if attributes.get("name") == "name":
                self.name = str(attributes.get("value", ""))
            if attributes.get("type") == "checkbox":
                self.choices[str(attributes["name"])] = str(attributes["value"])
                self.preselected += int("checked" in attributes)

    def handle_endtag(self, tag: str) -> None:
        super().handle_endtag(tag)
        if tag == "form":
            self._creation = False


def seed_demo(container: AppContainer) -> None:
    documents = [
        Document(document_id=identifier, title=title, source=source, text=text)
        for identifier, title, source, text in (
            (
                SELECTED_IDS[0],
                "Synthetic graph methods <note>",
                SOURCE,
                "Original synthetic methods, not a publication. Graph evidence needs human review.",
            ),
            (
                SELECTED_IDS[1],
                "Synthetic graph limitations",
                SOURCE,
                "Original synthetic limitations. A selected paper is not a scientific finding.",
            ),
            ("03-protocol", "Synthetic graph protocol (no chunks)", SOURCE, ""),
            ("04-excluded-source", "Synthetic graph elsewhere", "synthetic:other", "Other source."),
            ("05-excluded-title", "Synthetic unrelated note", SOURCE, "Different title."),
        )
    ]
    container.document_store.add_documents(
        documents,
        [
            Chunk(
                **document.model_dump(exclude={"metadata"}),
                chunk_id=f"synthetic-chunk-{index}",
                metadata={"chunk_index": "0"},
            )
            for index, document in enumerate(documents)
            if document.text
        ],
    )


@contextmanager
def creation_guards(container: AppContainer) -> Iterator[dict[str, Mock]]:
    with (
        browsing_guards(container, allow_collection_create=True) as guards,
        ExitStack() as stack,
    ):
        for name, target, method in (
            ("screening", container.paper_screening, "submit"),
            ("worksheets", container.worksheets, "build"),
            ("transitions", container.event_log, "append_transition"),
            ("socket_create", socket, "create_connection"),
            ("socket_connect", socket.socket, "connect"),
        ):
            guards[name] = stack.enter_context(
                patch.object(target, method, side_effect=AssertionError("Forbidden demo work."))
            )
        yield guards


def run_demo(output_dir: Path) -> str:
    """Save executed HTML, exact JSON, guard counts and a restartable synthetic database."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    database = output_dir / "corpus.sqlite3"
    settings = offline_settings(database)
    app = create_app(settings)
    container: AppContainer = app.state.container
    seed_demo(container)
    artifacts: dict[str, bytes] = {}
    pages: dict[str, str] = {}
    before_reads = database.read_bytes()
    with TestClient(app, root_path=PREFIX) as client, creation_guards(container) as guards:

        def get(name: str, url: str) -> CreationPage:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path.startswith(PREFIX + "/"):
                raise RuntimeError("Demo navigation left the local mounted app.")
            response = client.get(url)
            response.raise_for_status()
            if response.headers["cache-control"] != "no-store":
                raise RuntimeError("Native response lost no-store.")
            artifacts[name + ".html"] = response.content
            pages[name] = url
            return CreationPage(response.text)

        catalog = get(
            "catalog",
            PREFIX + "/explore?" + urlencode({"limit": 2, "source": SOURCE, "title": "graph"}),
        )
        listed = get("collections-before", catalog.links["saved-collections"])
        if listed.links["create-collection"] != catalog.links["create-collection"]:
            raise RuntimeError("Both native entry points must retain the catalog selection.")
        first = get("builder-first", listed.links["create-collection"])
        second = get("builder-next", first.links["next-page"])
        get_unchanged = database.read_bytes() == before_reads
        if (
            first.documents != list(SELECTED_IDS)
            or second.documents != ["03-protocol"]
            or first.name
            or second.name
            or first.preselected
            or second.preselected
            or tuple(json.loads(value) for value in first.choices.values()) != SELECTED_IDS
            or not get_unchanged
            or container.paper_collections.list_collections().collections
        ):
            raise RuntimeError(
                "Bounded catalog, blank controls, exact IDs or GET-only reads failed."
            )
        fields = {"name": NAME, **first.choices}
        with patch.object(
            container.paper_collections, "create", wraps=container.paper_collections.create
        ) as create:
            saved = client.post(
                first.form_action,
                data=fields,
                headers={"origin": "http://testserver", "sec-fetch-site": "same-origin"},
                follow_redirects=False,
            )
            if saved.status_code != 303:
                raise RuntimeError("Native create did not return 303.")
            location = saved.headers["location"]
            detail = get("saved-collection", location)
            records = container.paper_collections.list_collections().collections
            if len(records) != 1:
                raise RuntimeError("The explicit POST must save exactly one selection.")
            collection = container.paper_collections.get(records[0].collection_id)
            if collection.document_ids != SELECTED_IDS or collection.name != NAME:
                raise RuntimeError("Persisted membership differs from the checked exact IDs.")
            raw = client.get(PREFIX + f"/collections/{collection.collection_id}")
            raw.raise_for_status()
            artifacts["collection.json"] = raw.content
            api_equal = PaperCollection.model_validate_json(raw.content) == collection
            for name, link in (
                ("screening", "screen-member-1"),
                ("search", "search-collection"),
                ("worksheet", "worksheet-collection"),
            ):
                get(name, detail.links[link])
            before_rejections = database.read_bytes()
            rejected: dict[str, int] = {}
            for name, payload, origin in (
                ("duplicate-name", fields, "http://testserver"),
                ("empty-selection", {"name": "No papers"}, "http://testserver"),
                (
                    "unknown-document",
                    {"name": "Unknown member", **first.choices, "document_3": '"missing"'},
                    "http://testserver",
                ),
                ("cross-origin", fields, "https://external.invalid"),
            ):
                response = client.post(
                    first.form_action,
                    data=payload,
                    headers={"origin": origin},
                    follow_redirects=False,
                )
                rejected[name] = response.status_code
                artifacts[name + ".html"] = response.content
                if "location" in response.headers or database.read_bytes() != before_rejections:
                    raise RuntimeError("A rejected write changed the database or redirected.")
            attempts = create.call_count
        counts = {name: guard.call_count for name, guard in guards.items()}
    before_restart = database.read_bytes()
    restarted = create_app(settings)
    with (
        TestClient(restarted, root_path=PREFIX) as client,
        browsing_guards(restarted.state.container) as guards,
    ):
        response = client.get(location)
        response.raise_for_status()
        artifacts["restarted.html"] = response.content
        counts.update({f"restart:{name}": guard.call_count for name, guard in guards.items()})
    checks = {
        "fixture": FIXTURE,
        "corpus_documents": len(container.document_catalog.list_documents().documents),
        "filtered_page_sizes": [len(first.documents), len(second.documents)],
        "preselected_checkboxes": first.preselected + second.preselected,
        "get_database_unchanged": get_unchanged,
        "selected_document_ids": list(collection.document_ids),
        "selected_count": len(collection.document_ids),
        "collection_count": len(records),
        "revision": collection.revision,
        "redirect_status": saved.status_code,
        "create_service_attempts": attempts,
        "rejected_statuses": rejected,
        "rejected_writes_database_unchanged": database.read_bytes() == before_rejections,
        "python_json_equal": api_equal,
        "restart_selection_equal": SQLitePaperCollections(database).get(collection.collection_id)
        == collection,
        "restart_html_identical": artifacts["restarted.html"] == artifacts["saved-collection.html"],
        "restart_database_unchanged": database.read_bytes() == before_restart,
        "handoff_pages": ["screening", "search", "worksheet"],
        "events": len(container.event_log.list_events()),
        "forbidden_call_counts": counts,
    }
    if (
        rejected
        != {
            "duplicate-name": 409,
            "empty-selection": 422,
            "unknown-document": 422,
            "cross-origin": 403,
        }
        or attempts != 3
        or checks["events"] != 0
        or any(counts.values())
        or not all(
            checks[key]
            for key in (
                "rejected_writes_database_unchanged",
                "python_json_equal",
                "restart_selection_equal",
                "restart_html_identical",
                "restart_database_unchanged",
            )
        )
    ):
        raise RuntimeError("Executed collection creation checks failed.")
    transcript = (
        f"{PANEL_TITLES[0]}\n"
        f"Synthetic corpus: {checks['corpus_documents']} papers; filtered page sizes: "
        f"{checks['filtered_page_sizes']}.\n"
        f"Preselected checkboxes: {checks['preselected_checkboxes']}; GET database unchanged.\n"
        "Both catalog and saved-collection links open the same bounded builder.\n\n"
        f"{PANEL_TITLES[1]}\n"
        f"Checked IDs saved: {checks['selected_count']}; "
        f"collection count: {checks['collection_count']}.\n"
        f"Native POST: {checks['redirect_status']}; persisted revision: {checks['revision']}.\n"
        "Slash, CRLF, NUL and Unicode IDs round-trip; the later page is not selected.\n\n"
        f"{PANEL_TITLES[2]}\n"
        f"Duplicate name: {rejected['duplicate-name']}; empty: {rejected['empty-selection']}; "
        f"unknown ID: {rejected['unknown-document']}; cross-origin: {rejected['cross-origin']}.\n"
        "Rejected writes leave database bytes unchanged: "
        f"{checks['rejected_writes_database_unchanged']}.\n"
        f"Create service attempts: {attempts}; one successful collection, no partial selection.\n\n"
        f"{PANEL_TITLES[3]}\n"
        f"Python/JSON equal: {api_equal}; "
        f"restart HTML identical: {checks['restart_html_identical']}.\n"
        f"Opened handoffs: {', '.join(checks['handoff_pages'])}.\n"
        f"Agent events: {checks['events']}; forbidden calls: {sum(counts.values())}.\n"
        "Executed synthetic HTTP measurements, not scientific findings "
        "or a performance benchmark.\n"
    )
    for filename, value in (("checks.json", checks), ("pages.json", pages)):
        artifacts[filename] = (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode()
    for filename, content in artifacts.items():
        (output_dir / filename).write_bytes(content)
    return transcript


def serve_demo(output_dir: Path, port: int) -> None:
    """Serve the explicit synthetic fixture on loopback, allowing native collection creation."""
    database = output_dir / "corpus.sqlite3"
    checks = json.loads((output_dir / "checks.json").read_text(encoding="utf-8"))
    if checks.get("fixture") != FIXTURE or not database.is_file():
        raise ValueError("Expected an existing native collection creation demo directory.")
    app = create_app(offline_settings(database))
    mounted = FastAPI()
    mounted.mount(PREFIX, app)
    with browsing_guards(app.state.container, allow_collection_create=True):
        uvicorn.run(mounted, host="127.0.0.1", port=port, access_log=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output-dir", type=Path, help="Create a new measured synthetic demo.")
    action.add_argument("--serve", type=Path, help="Serve that existing demo on loopback only.")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    if arguments.serve:
        serve_demo(arguments.serve, arguments.port)
    else:
        print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
