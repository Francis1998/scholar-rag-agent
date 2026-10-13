"""Execute a native worksheet form against isolated original synthetic notes."""

import argparse
import asyncio
import json
import socket
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import httpx
from fastapi.testclient import TestClient

from agent.research_worksheet import ResearchWorksheet, WorksheetRequest
from api.application import create_app
from api.dependencies import AppContainer
from llm.fake import FakeLLMAdapter
from llm.router import RoutingLLMAdapter
from retrieval.models import Chunk, Document
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage

SELECTED_ID = "paper/../synthetic-\u7814?draft#v1"
QUESTIONS = ("What retrieval methods are described?", "What limitations should I inspect?")
PANEL_TITLES = (
    "1. Open a native worksheet from the corpus",
    "2. Inspect every question / paper cell",
    "3. Follow exact current-source links",
    "4. Measure the read-only boundary",
)


class WorksheetPage(SearchPage):
    """Read native controls from actual returned HTML, including textarea semantics."""

    def __init__(self, html: str) -> None:
        self._textarea = ""
        super().__init__(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        if tag == "textarea":
            self._textarea = str(dict(attrs)["name"])
            self.form_values[self._textarea] = ""

    def handle_data(self, data: str) -> None:
        super().handle_data(data)
        if self._textarea:
            self.form_values[self._textarea] += data

    def handle_endtag(self, tag: str) -> None:
        super().handle_endtag(tag)
        if tag == "textarea":
            self.form_values[self._textarea] = self.form_values[self._textarea].removeprefix("\n")
            self._textarea = ""


def _deny(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Worksheet inspection attempted generation, external HTTP or a write.")


@contextmanager
def worksheet_guards(container: AppContainer) -> Iterator[dict[str, Mock]]:
    """Permit actual retrieval, but turn forbidden work into a measured failure."""
    with ExitStack() as stack:
        guards = {}
        for target, method in (
            (FakeLLMAdapter, "generate"),
            (RoutingLLMAdapter, "generate"),
            (container.llm, "generate"),
            (container.runner, "run"),
            (container.runner._executor, "answer"),
            (container.runner._executor._grounder, "ground"),
            (container.document_store, "add_documents"),
            (container.hybrid_retriever, "add_chunks"),
            (container.graph_builder, "index_chunks"),
            (container.ingestion_pipeline, "ingest_documents"),
            (container.event_log, "append_event"),
            (container.event_log, "append_transition"),
            (container.paper_collections, "create"),
            (container.paper_collections, "replace"),
            (container.paper_collections, "delete"),
        ):
            name = f"{getattr(target, '__name__', type(target).__name__)}.{method}"
            guards[name] = stack.enter_context(patch.object(target, method, side_effect=_deny))
        yield guards


def _get(client: TestClient, url: str) -> httpx.Response:
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path.startswith("/explore"):
        raise RuntimeError("Worksheet navigation left the local explorer.")
    response = client.get(url)
    response.raise_for_status()
    return response


def run_demo(output_dir: Path) -> str:
    """Save executed HTML/downloads and measurements; do not retain a corpus or start a server."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    with TemporaryDirectory(prefix="browser-worksheet-") as temporary, ExitStack() as stack:
        network = [
            stack.enter_context(patch.object(target, method, side_effect=_deny))
            for target, method in (
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
                (socket, "create_connection"),
                (socket.socket, "connect"),
            )
        ]
        database = Path(temporary) / "corpus.sqlite3"
        settings = offline_settings(database).model_copy(
            update={"max_source_docs": 1, "max_hops": 1}
        )
        app = create_app(settings)
        container: AppContainer = app.state.container
        chunks = [
            Chunk(
                document_id=identifier,
                chunk_id=f"{identifier}/chunk-0",
                title=title,
                source="synthetic:browser-worksheet",
                text=text,
                metadata={"chunk_index": "0"},
            )
            for identifier, title, text in (
                (
                    SELECTED_ID,
                    "Synthetic lexical methods",
                    "Original synthetic note, not a publication or result. Retrieval uses terms. "
                    "Literal <script>not executable</script> & cafe\u0301 \u7814\u7a76. "
                    + "A score does not establish scientific support. "
                    * 25,
                ),
                (
                    "second-paper",
                    "Synthetic graph limitations",
                    "Original synthetic note. Graph retrieval follows co-mentions; "
                    "connections are not causal or scientific conclusions.",
                ),
                ("unselected", "Excluded synthetic note", "EXCLUDED_MARKER retrieval limitations."),
            )
        ]
        container.document_store.add_documents(
            [Document(**chunk.model_dump(exclude={"chunk_id"})) for chunk in chunks], chunks
        )
        container.hybrid_retriever.add_chunks(chunks)
        container.graph_builder.index_chunks(chunks)
        collection = container.paper_collections.create(
            name="Synthetic worksheet selection", document_ids=[SELECTED_ID, "second-paper"]
        )
        before = database.read_bytes()
        with TestClient(app) as client, worksheet_guards(container) as guards:
            catalog = _get(client, "/explore?limit=2")
            opened = _get(client, WorksheetPage(catalog.text).links["research-worksheet"])
            document = _get(client, WorksheetPage(catalog.text).links["inspect-document-1"])
            prefilled = _get(client, WorksheetPage(document.text).links["worksheet-document"])
            if json.loads(WorksheetPage(prefilled.text).form_values["document_ids"]) != [
                SELECTED_ID
            ]:
                raise RuntimeError("Native document prefill did not retain the exact ID.")
            page = WorksheetPage(opened.text)
            fields = {
                **page.form_values,
                "questions": "\r\n".join(QUESTIONS),
                "document_ids": json.dumps([SELECTED_ID, "second-paper"]),
                "passages_per_cell": "1",
            }
            html = client.post(page.form_action, data=fields)
            html.raise_for_status()
            result_page = WorksheetPage(html.text)
            source_url = result_page.links["source-1-1-1"]
            source = _get(client, source_url)
            if result_page.links["back-to-catalog"] != "/explore?limit=2":
                raise RuntimeError("Native return navigation lost its catalog selection.")
            artifacts = {
                "catalog.html": catalog.content,
                "form.html": prefilled.content,
                "worksheet.html": html.content,
                "source.html": source.content,
            }
            for fmt, extension in (("json", "json"), ("markdown", "md")):
                response = client.post(page.form_action, data={**fields, "format": fmt})
                response.raise_for_status()
                artifacts["worksheet." + extension] = response.content
            worksheet = ResearchWorksheet.model_validate_json(artifacts["worksheet.json"])
            payload = WorksheetRequest(
                questions=QUESTIONS, document_ids=(SELECTED_ID, "second-paper"), passages_per_cell=1
            )
            api_response = client.post(
                "/research/worksheet", json=payload.model_dump(mode="json", exclude_none=True)
            )
            api_response.raise_for_status()
            python_result = asyncio.run(container.worksheets.build(payload))
            selected = client.post(
                page.form_action,
                data={
                    **fields,
                    "document_ids": "",
                    "collection_id": collection.collection_id,
                    "format": "json",
                },
            )
            selected.raise_for_status()
            collection_result = ResearchWorksheet.model_validate_json(selected.content)
            invalid = client.post(
                page.form_action, data={**fields, "document_ids": "", "collection_id": ""}
            )
            counts = {name: guard.call_count for name, guard in guards.items()}
        if database.read_bytes() != before:
            raise RuntimeError("Worksheet reads changed database bytes.")
        restarted = create_app(settings)
        with TestClient(restarted) as client, worksheet_guards(restarted.state.container) as guards:
            reopened = client.post(page.form_action, data=fields)
            reopened.raise_for_status()
            counts.update({f"restart:{name}": guard.call_count for name, guard in guards.items()})
        checks = {
            "questions": len(worksheet.rows),
            "papers": len(worksheet.documents),
            "cells": sum(len(row.cells) for row in worksheet.rows),
            "native_prefill_and_return": True,
            "source_context_exact_id": WorksheetPage(source.text).chunks == [chunks[0].chunk_id],
            "source_url": source_url,
            "excerpt_truncated": worksheet.rows[0].cells[0].passages[0].excerpt_truncated,
            "empty_selection_status": invalid.status_code,
            "api_python_browser_equal": (
                python_result == worksheet
                and api_response.content == artifacts["worksheet.json"]
                and worksheet.to_markdown().encode() == artifacts["worksheet.md"]
            ),
            "collection_same_cells": collection_result.rows == worksheet.rows,
            "restart_html_identical": reopened.content == html.content,
            "database_bytes_unchanged": database.read_bytes() == before,
            "agent_events": len(container.event_log.list_events()),
            "forbidden_call_counts": counts,
            "external_connection_calls": sum(guard.call_count for guard in network),
            "output_bytes": {name: len(content) for name, content in artifacts.items()},
        }
        if (
            any(counts.values())
            or checks["external_connection_calls"] != 0
            or checks["agent_events"] != 0
            or checks["cells"] != 4
            or invalid.status_code != 422
            or "EXCLUDED_MARKER" in html.text
            or not all(value for value in checks.values() if isinstance(value, bool))
        ):
            raise RuntimeError("The measured native worksheet or read-only contract failed.")
    checks["temporary_database_removed"] = not database.exists()
    if not checks["temporary_database_removed"]:
        raise RuntimeError("Temporary fixture database remained.")
    transcript = (
        "\n\n".join(
            (
                f"{PANEL_TITLES[0]}\n"
                "Executed catalog -> paper -> prefilled form; exact Unicode / slash ID retained.\n"
                "Native POST form; no JavaScript, provider request, or saved worksheet.\n"
                "The catalog return link preserves its original page size.",
                f"{PANEL_TITLES[1]}\n"
                f"Questions: {checks['questions']}; papers: {checks['papers']}; "
                f"cells: {checks['cells']}.\n"
                f"Excerpt truncation displayed: {checks['excerpt_truncated']} (800 characters).\n"
                "Every cell exposes source IDs, score, path and full-text digest.\n"
                "Retrieval passages are not answers, confidence estimates or findings.",
                f"{PANEL_TITLES[2]}\n"
                f"Exact current-source context: {checks['source_context_exact_id']}.\n"
                f"API / Python / browser exports equal: {checks['api_python_browser_equal']}.\n"
                f"Collection resolves the same cells: {checks['collection_same_cells']}.\n"
                "Each export submission reruns retrieval; it is not a frozen download.",
                f"{PANEL_TITLES[3]}\n"
                f"Forbidden model/write calls: {sum(counts.values())}; "
                f"agent events: {checks['agent_events']}.\n"
                f"External connections: {checks['external_connection_calls']}; "
                f"empty selection: HTTP {checks['empty_selection_status']}.\n"
                f"Database unchanged: {checks['database_bytes_unchanged']}; "
                f"unchanged-corpus restart identical: {checks['restart_html_identical']}.\n"
                f"HTML: {len(html.content)} bytes; JSON: {len(artifacts['worksheet.json'])}; "
                f"Markdown: {len(artifacts['worksheet.md'])}.\n"
                "Temporary corpus removed. Original synthetic data; no scientific results.",
            )
        )
        + "\n"
    )
    artifacts["checks.json"] = (json.dumps(checks, ensure_ascii=True, indent=2) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode()
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as target:
            target.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(run_demo(parser.parse_args().output_dir), end="")


if __name__ == "__main__":
    main()
