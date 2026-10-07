"""Exercise the live HTML explorer using only a new, persistent synthetic corpus."""

import argparse
import hashlib
import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from html.parser import HTMLParser
from pathlib import Path
from typing import NoReturn
from unittest.mock import Mock, patch
from urllib.parse import urlencode, urlsplit

import httpx
import uvicorn
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_evidence_export import offline_settings

FIXTURE = "scholar-corpus-explorer-synthetic-v1"
SELECTED_ID = "01-paper/../\u7814\u7a76?draft#v1"
SOURCE = "synthetic:explorer"
FRAME_PAGES = (
    "catalog-first",
    "catalog-next",
    "passages-first",
    "passages-next",
)


class ExplorerPage(HTMLParser):
    """Inspect actual navigation and identity text, without executing HTML."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.links: dict[str, str] = {}
        self.documents: list[str] = []
        self.chunks: list[str] = []
        self.passages: list[str] = []
        self.indices: list[str] = []
        self._field = ""
        self._tag = ""
        self._text = ""
        self.feed(html)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("id") and attributes.get("href"):
            self.links[str(attributes["id"])] = str(attributes["href"])
        if tag == "pre" or (tag == "dd" and attributes.get("class") == "chunk-index"):
            self._field = str(attributes.get("class", ""))
            self._tag = tag
            self._text = ""

    def handle_data(self, data: str) -> None:
        if self._field:
            self._text += data

    def handle_endtag(self, tag: str) -> None:
        if tag != self._tag:
            return
        if "document-id" in self._field.split():
            self.documents.append(json.loads(self._text))
        if "chunk-id" in self._field.split():
            self.chunks.append(json.loads(self._text))
        if "passage-text" in self._field.split():
            self.passages.append(self._text)
        if self._field == "chunk-index":
            self.indices.append(self._text)
        self._field = ""
        self._tag = ""


def seed_demo(container: AppContainer) -> None:
    """Store original synthetic imported fixtures, including a non-path-safe ID."""
    documents = [
        Document(
            document_id=identifier,
            title=title,
            source=source,
            text="UNEXPOSED_DOCUMENT_BODY",
            metadata={"private": "UNEXPOSED_METADATA"},
        )
        for identifier, title, source in (
            (SELECTED_ID, "Synthetic graph methods", SOURCE),
            ("02-limitations", "Synthetic graph limitations", SOURCE),
            ("03-protocol", "Synthetic graph protocol (no chunks)", SOURCE),
            ("04-excluded-source", "Synthetic graph from another source", "synthetic:excluded"),
            ("05-excluded-title", "Synthetic unrelated note", SOURCE),
        )
    ]
    selected = documents[0]
    chunks = [
        Chunk(
            chunk_id=identifier,
            document_id=selected.document_id,
            title=selected.title,
            source=selected.source,
            text=text,
            metadata={"chunk_index": str(index), "private": "UNEXPOSED_METADATA"},
        )
        for identifier, index, text in (
            (
                "chunk-1",
                0,
                "Synthetic fixture, not research findings.\n"
                "Inspect a passage before choosing evidence. A source ID is not proof.\n"
                'Literal markup stays text: <script>alert("fixture")</script>',
            ),
            (
                "chunk-10",
                9,
                "Synthetic tenth chunk. Its ID sorts before chunk-2, even though "
                "its stored ordinal is 9. This browser reads stored text, not ranked answers.",
            ),
            (
                "chunk-2",
                1,
                "Synthetic second chunk. The exact paper ID includes Unicode, slashes "
                "and a dot segment; query encoding preserves its scope across pages.",
            ),
            (
                "chunk-20",
                19,
                "Synthetic twentieth chunk. These pages never run a model, retrieve, "
                "rerank or append events. Restart with the same database to keep browsing.",
            ),
            ("chunk-3", 2, "Synthetic oversized imported passage. " * 130),
        )
    ]
    for document in documents[1:]:
        if document.document_id != "03-protocol":
            chunks.append(
                Chunk(
                    chunk_id=document.document_id + "-chunk",
                    document_id=document.document_id,
                    title=document.title,
                    source=document.source,
                    text="EXCLUDED_EVIDENCE: a different synthetic document.",
                )
            )
    container.document_store.add_documents(documents, chunks)


def _no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError(
        "Explorer demo attempted generation, retrieval, mutation or external HTTP."
    )


@contextmanager
def browsing_guards(container: AppContainer) -> Iterator[dict[str, Mock]]:
    """Make forbidden work a failing measurement, not an assumption about a fake provider."""
    with ExitStack() as stack:
        guards = {}
        for target, method in (
            (container.document_store, "list_chunks"),
            (container.document_store, "add_documents"),
            (container.hybrid_retriever, "retrieve"),
            (container.hybrid_retriever, "add_chunks"),
            (container.runner._executor._reranker, "rerank"),
            (container.llm, "generate"),
            (container.runner, "run"),
            (container.runner, "preview"),
            (container.event_log, "append_event"),
            (container.ingestion_pipeline, "ingest_documents"),
            (container.paper_collections, "create"),
            (container.paper_collections, "replace"),
            (container.paper_collections, "delete"),
            (httpx.HTTPTransport, "handle_request"),
            (httpx.AsyncHTTPTransport, "handle_async_request"),
        ):
            name = f"{getattr(target, '__name__', type(target).__name__)}.{method}"
            guards[name] = stack.enter_context(patch.object(target, method, side_effect=_no_work))
        yield guards


def _get(client: TestClient, url: str) -> tuple[httpx.Response, ExplorerPage]:
    parts = urlsplit(url)
    if parts.netloc or parts.scheme or parts.path not in {"/explore", "/explore/document"}:
        raise RuntimeError("Explorer navigation escaped its local GET routes.")
    response = client.get(url)
    response.raise_for_status()
    if response.headers["cache-control"] != "no-store" or "UNEXPOSED_" in response.text:
        raise RuntimeError("Explorer response leaked metadata or lost its privacy headers.")
    return response, ExplorerPage(response.text)


def run_demo(output_dir: Path) -> str:
    """Persist the primary demo plus measured HTML, checks, and reproducible browser paths."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    path = output_dir / "corpus.sqlite3"
    settings = offline_settings(path)
    app = create_app(settings)
    seed_demo(app.state.container)
    before = path.read_bytes()
    events_before = app.state.container.event_log.list_events()
    urls = {
        "catalog-first": "/explore?" + urlencode({"limit": 2, "source": SOURCE, "title": "graph"})
    }
    responses: dict[str, httpx.Response] = {}
    with TestClient(app) as client, browsing_guards(app.state.container) as guards:
        responses["catalog-first"], first = _get(client, urls["catalog-first"])
        urls["catalog-next"] = first.links["next-page"]
        responses["catalog-next"], second = _get(client, urls["catalog-next"])
        if first.documents != [SELECTED_ID, "02-limitations"] or second.documents != [
            "03-protocol"
        ]:
            raise RuntimeError("Filtered HTML catalog pages did not preserve source/title scope.")
        urls["passages-first"] = first.links["inspect-document-1"]
        pages = []
        for name in ("passages-first", "passages-next", "passages-last"):
            responses[name], page = _get(client, urls[name])
            pages.append(page)
            next_name = {"passages-first": "passages-next", "passages-next": "passages-last"}.get(
                name
            )
            if next_name:
                urls[next_name] = page.links["next-page"]
        if (
            [page.chunks for page in pages]
            != [
                ["chunk-1", "chunk-10"],
                ["chunk-2", "chunk-20"],
                ["chunk-3"],
            ]
            or any(page.documents != [SELECTED_ID] for page in pages)
            or [int(index) for page in pages for index in page.indices] != [0, 9, 1, 19, 2]
            or "next-page" in pages[-1].links
            or any("EXCLUDED_EVIDENCE" in text for page in pages for text in page.passages)
            or len(pages[-1].passages[0]) != 4000
            or "Passage truncated" not in responses["passages-last"].text
            or pages[1].links["back-to-catalog"] != urls["catalog-first"]
        ):
            raise RuntimeError("HTML passage pagination, exact identity or truncation failed.")
        _, blank = _get(client, "/explore?source=&title=&limit=20")
        if len(blank.documents) != 5:
            raise RuntimeError("Native blank form fields did not mean omitted filters.")
        guard_counts = {name: guard.call_count for name, guard in guards.items()}

    restarted = create_app(settings)
    with TestClient(restarted) as client, browsing_guards(restarted.state.container) as guards:
        for name, url in urls.items():
            response, _ = _get(client, url)
            if response.content != responses[name].content:
                raise RuntimeError("HTML changed after restarting against the same stored corpus.")
        for name, guard in guards.items():
            guard_counts[name] += guard.call_count
    events_after = restarted.state.container.event_log.list_events()
    if path.read_bytes() != before or events_before != events_after or any(guard_counts.values()):
        raise RuntimeError(
            "Browsing or restart changed the database/events or attempted forbidden work."
        )
    checks = {
        "fixture": FIXTURE,
        "selected_document_id": SELECTED_ID,
        "catalog_pages": [first.documents, second.documents],
        "chunk_pages": [page.chunks for page in pages],
        "chunk_indices_in_id_order": [int(index) for page in pages for index in page.indices],
        "last_passage_characters": len(pages[-1].passages[0]),
        "last_passage_truncated": True,
        "restart_html_identical": True,
        "database_bytes_unchanged": True,
        "database_sha256": hashlib.sha256(before).hexdigest(),
        "events_before": len(events_before),
        "events_after": len(events_after),
        "forbidden_call_counts": guard_counts,
    }
    for name, response in responses.items():
        (output_dir / f"{name}.html").write_bytes(response.content)
    for filename, value in (("checks.json", checks), ("pages.json", urls)):
        (output_dir / filename).write_text(
            json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
    transcript = (
        "Local corpus explorer / original synthetic fixtures / offline\n"
        "Filtered catalog pages: 2, 1 (exact source + literal title)\n"
        "Native blank source/title fields: 5 papers, no filters\n"
        "Selected document ID (JSON): " + json.dumps(SELECTED_ID) + "\n"
        "Passage pages: 2, 2, 1; exact document scope; final cursor absent\n"
        "Chunk IDs: chunk-1, chunk-10, chunk-2, chunk-20, chunk-3\n"
        "Stored indices: 0, 9, 1, 19, 2 (not passage order)\n"
        "Imported final passage: 4000 characters; explicit truncation notice\n"
        f"Events before/after: {len(events_before)}/{len(events_after)}\n"
        "Forbidden generation/retrieval/reranking/writes/external HTTP calls: 0\n"
        "Restart: all five HTML pages identical; SQLite bytes unchanged\n"
        "The synthetic corpus persists for the optional loopback browser demo.\n"
        "This is an inspection workflow, not scientific validation or a frozen run export.\n"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def serve_demo(output_dir: Path, port: int) -> None:
    """Serve only an explicitly generated fixture, with ambient credentials ignored."""
    path = output_dir / "corpus.sqlite3"
    checks = json.loads((output_dir / "checks.json").read_text(encoding="utf-8"))
    if checks.get("fixture") != FIXTURE or not path.is_file():
        raise ValueError("Expected an existing corpus explorer demo directory.")
    app = create_app(offline_settings(path))
    before = path.read_bytes()
    with browsing_guards(app.state.container):
        uvicorn.run(app, host="127.0.0.1", port=port, access_log=False)
    if path.read_bytes() != before:
        raise RuntimeError("The synthetic database changed during browser inspection.")


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
