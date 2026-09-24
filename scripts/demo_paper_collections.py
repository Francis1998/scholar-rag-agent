"""Measure saved paper selections through the real API using only synthetic offline data."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from agent.retrieval_preview import RetrievalPreview
from api.application import create_app
from api.dependencies import AppContainer
from llm.fake import FakeLLMAdapter
from llm.schemas import LLMRequest, LLMResponse
from scripts.demo_evidence_export import offline_settings
from storage.paper_collections import CollectionPage, PaperCollection

QUERY = "retrieval"
PANEL_TITLES = (
    "1. Save named selections, not copies of papers",
    "2. Reopen SQLite and reuse a saved selection",
    "3. Replace membership without losing old evidence",
    "4. Delete metadata, keep papers and run history",
)
ARTIFACT_NAMES = (
    "collection.json",
    "page-1.json",
    "page-2.json",
    "preview-original.json",
    "preview-updated.json",
    "replacement.json",
    "evidence.json",
    "evidence.md",
    "documents-after.json",
    "checks.json",
    "transcript.txt",
)


def _no_network(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("The synthetic collection demo forbids external HTTP.")


def _preview(client: TestClient, collection_id: str) -> RetrievalPreview:
    response = client.post("/retrieve", json={"query": QUERY, "collection_id": collection_id})
    response.raise_for_status()
    return RetrievalPreview.model_validate_json(response.content)


def run_demo(output_dir: Path) -> str:
    """Save measured results; ignore environment/dotenv and clean up isolated SQLite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        target = output_dir / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}; choose a new directory.")
    calls = 0
    generate = FakeLLMAdapter.generate

    async def counted_fake(adapter: FakeLLMAdapter, request: LLMRequest) -> LLMResponse:
        nonlocal calls
        calls += 1
        return await generate(adapter, request)

    with (
        TemporaryDirectory(prefix="scholar-paper-collections-") as temporary,
        patch.object(httpx.HTTPTransport, "handle_request", _no_network),
        patch.object(httpx.AsyncHTTPTransport, "handle_async_request", _no_network),
        patch.object(FakeLLMAdapter, "generate", counted_fake),
    ):
        database = Path(temporary) / "corpus.sqlite3"
        settings = offline_settings(database)
        application = create_app(settings)
        with TestClient(application) as client:
            identifiers = []
            for title, marker in (
                ("Synthetic methods", "METHODS_MARKER"),
                ("Synthetic comparison", "COMPARISON_MARKER"),
                ("Synthetic background", "EXCLUDED_MARKER"),
            ):
                response = client.post(
                    "/ingest/text",
                    json={
                        "title": title,
                        "source": "synthetic:collections",
                        "text": f"Synthetic data, not a publication. Local retrieval {marker}.",
                    },
                )
                response.raise_for_status()
                identifiers.append(response.json()["document_id"])
            created = client.post(
                "/collections",
                json={
                    "name": "  Review methods  ",
                    "document_ids": [identifiers[0], identifiers[1], identifiers[0]],
                },
            )
            created.raise_for_status()
            collection = PaperCollection.model_validate_json(created.content)
            client.post(
                "/collections",
                json={
                    "name": "Background",
                    "document_ids": [identifiers[2]],
                },
            ).raise_for_status()
            page_one = client.get("/collections", params={"limit": 1})
            page_one.raise_for_status()
            first = CollectionPage.model_validate_json(page_one.content)
            if first.next_cursor is None:
                raise RuntimeError("The demo needs two collection pages.")
            page_two = client.get("/collections", params={"limit": 1, "cursor": first.next_cursor})
            page_two.raise_for_status()
            second = CollectionPage.model_validate_json(page_two.content)
            if (
                len(first.collections) != 1
                or len(second.collections) != 1
                or second.next_cursor
                or first.collections[0].collection_id == second.collections[0].collection_id
                or collection.document_ids != tuple(identifiers[:2])
            ):
                raise RuntimeError("Collection paging or normalized membership did not match.")
            initial_preview = _preview(client, collection.collection_id)
            container: AppContainer = application.state.container
            idle_events = len(container.event_log.list_events())
            idle_calls = calls
            if idle_calls or idle_events:
                raise RuntimeError("CRUD and preview must not call generators or write events.")

        reopened = create_app(settings)
        container = reopened.state.container
        path = f"/collections/{collection.collection_id}"
        with TestClient(reopened) as client:
            after_restart = client.get(path)
            after_restart.raise_for_status()
            unchanged = after_restart.content == created.content
            preview = _preview(client, collection.collection_id)
            if not unchanged or preview != initial_preview:
                raise RuntimeError("Collection metadata and preview must survive restart.")
            if {source.chunk.document_id for source in preview.sources} != set(
                identifiers[:2]
            ) or "EXCLUDED_MARKER" in preview.context:
                raise RuntimeError("Preview escaped the saved selection.")
            query = client.post(
                "/query",
                json={
                    "query": QUERY,
                    "collection_id": collection.collection_id,
                },
            )
            query.raise_for_status()
            result = query.json()["result"]
            if result["state"] != "DONE":
                raise RuntimeError("The synthetic fake-model query did not finish.")
            export_path = f"/runs/{result['run_id']}/export"
            exported = client.get(export_path)
            markdown = client.get(export_path, params={"format": "markdown"})
            exported.raise_for_status()
            markdown.raise_for_status()
            evidence = EvidenceBundle.model_validate_json(exported.content)
            if (
                evidence.generation.provider != "fake"
                or evidence.snapshot.sources != preview.sources
                or evidence.snapshot.request.context != preview.context
            ):
                raise RuntimeError("The fake query did not preserve the exact preview evidence.")
            events_before = container.event_log.list_events()
            replacement = client.put(
                path,
                json={
                    "name": "Review methods",
                    "document_ids": [identifiers[1]],
                    "expected_revision": 1,
                },
            )
            replacement.raise_for_status()
            replaced = PaperCollection.model_validate_json(replacement.content)
            stale = client.put(
                path,
                json={
                    "name": "Stale edit",
                    "document_ids": [identifiers[0]],
                    "expected_revision": 1,
                },
            )
            updated_preview = _preview(client, collection.collection_id)
            if (
                replaced.revision != 2
                or stale.status_code != 409
                or {source.chunk.document_id for source in updated_preview.sources}
                != {identifiers[1]}
            ):
                raise RuntimeError("Replacement, stale-write protection, or new scope failed.")
            removed = client.delete(path, params={"expected_revision": 2})
            removed.raise_for_status()
            missing = client.post(
                "/query",
                json={
                    "query": QUERY,
                    "collection_id": collection.collection_id,
                },
            )
            if missing.status_code != 404 or removed.status_code != 204:
                raise RuntimeError("Deleted collections must never fall back to the corpus.")

        restarted_again = create_app(settings)
        with TestClient(restarted_again) as client:
            json_after = client.get(export_path)
            markdown_after = client.get(export_path, params={"format": "markdown"})
            documents = client.get("/documents")
            runs = client.get("/runs")
            for response in (json_after, markdown_after, documents, runs):
                response.raise_for_status()
            same_json = json_after.content == exported.content
            same_markdown = markdown_after.content == markdown.content
            event_delta = len(restarted_again.state.container.event_log.list_events()) - len(
                events_before
            )
            paper_count = len(documents.json()["documents"])
            run_count = len(runs.json()["runs"])
            if (
                not same_json
                or not same_markdown
                or event_delta
                or paper_count != 3
                or run_count != 1
            ):
                raise RuntimeError("Metadata edits/deletion changed corpus or historical evidence.")
        if calls != 1:
            raise RuntimeError("Only the single explicit /query may call the offline generator.")
    database_removed = not database.exists()
    if not database_removed:
        raise RuntimeError("Temporary SQLite was not removed.")
    checks = {
        "synthetic_only": True,
        "crud_preview_generation_calls": idle_calls,
        "crud_preview_events": idle_events,
        "total_fake_generation_calls": calls,
        "collection_restart_identical": unchanged,
        "preview_restart_identical": preview == initial_preview,
        "original_document_ids": list(collection.document_ids),
        "updated_document_ids": list(replaced.document_ids),
        "original_source_count": len(preview.sources),
        "updated_source_count": len(updated_preview.sources),
        "stale_update_status": stale.status_code,
        "deleted_query_status": missing.status_code,
        "json_export_unchanged": same_json,
        "markdown_export_unchanged": same_markdown,
        "post_query_event_delta": event_delta,
        "remaining_documents": paper_count,
        "remaining_runs": run_count,
        "temporary_database_removed": database_removed,
    }
    artifacts = {
        "collection.json": created.content,
        "page-1.json": page_one.content,
        "page-2.json": page_two.content,
        "preview-original.json": preview.model_dump_json(indent=2).encode(),
        "preview-updated.json": updated_preview.model_dump_json(indent=2).encode(),
        "replacement.json": replacement.content,
        "evidence.json": exported.content,
        "evidence.md": markdown.content,
        "documents-after.json": documents.content,
        "checks.json": (json.dumps(checks, indent=2, sort_keys=True) + "\n").encode(),
    }
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"Synthetic corpus: {len(identifiers)} papers; saved collections: 2",
                        f"Name: {collection.name}; unique members: {len(collection.document_ids)}",
                        f"GET /collections?limit=1 -> {len(first.collections)}"
                        f" + {len(second.collections)} rows",
                        f"CRUD/preview generator calls: {idle_calls}; agent events: {idle_events}",
                        "Selections reference existing IDs; no corpus text is copied.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"Metadata after restart identical: {unchanged}",
                        f"Preview after restart identical: {preview == initial_preview}",
                        f"POST /retrieve selected passages: {len(preview.sources)}",
                        f"POST /query state: {evidence.status}; "
                        f"provider: {evidence.generation.provider}",
                        "Query evidence matches preview; excluded background passages: 0",
                        "Fake output demonstrates plumbing, not research findings.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"PUT revision {collection.revision} -> {replaced.revision}",
                        f"Current selection: {len(replaced.document_ids)} paper; "
                        f"preview: {len(updated_preview.sources)}",
                        f"Stale revision rejected: HTTP {stale.status_code}",
                        f"Historical export retains {len(evidence.snapshot.sources)} "
                        "original passages.",
                        f"JSON unchanged: {same_json}; Markdown unchanged: {same_markdown}",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"DELETE -> {removed.status_code}; "
                        f"subsequent scoped query -> {missing.status_code}",
                        f"Remaining corpus papers: {paper_count}; saved runs: {run_count}",
                        f"Extra events from edits, preview, delete, export: {event_delta}",
                        f"Total model calls: {calls} (the explicit fake query only)",
                        f"Temporary SQLite removed: {database_removed}",
                        "No access control or source-content snapshot; review saved artifacts.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("paper-collections-demo"))
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
