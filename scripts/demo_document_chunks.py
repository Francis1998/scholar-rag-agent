"""Inspect actual persisted evidence pages using only isolated synthetic/offline data."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn
from unittest.mock import patch
from urllib.parse import quote

from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_evidence_export import offline_settings
from storage.document_catalog import DocumentCatalogPage
from storage.document_chunks import DocumentChunksPage

PANEL_TITLES = (
    "1. Save a synthetic corpus, then restart",
    "2. Discover a paper and inspect its chunks",
    "3. Continue by exact stored chunk identity",
    "4. Inspect bounds before selecting evidence",
)
_TEXT = (
    "Synthetic data only, not scientific findings. "
    "GraphRAG connects synthetic methods and limitations. "
    "Review the stored passage before choosing a query scope. "
) * 24
_OUTPUTS = (
    "catalog.json",
    "page-1.json",
    "page-2.json",
    "page-3.json",
    "long-chunk.json",
    "query-request.json",
    "transcript.txt",
)


def _no_work(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Evidence browsing attempted retrieval, generation, or mutation.")


def run_demo(output_dir: Path) -> str:
    """Render real API outcomes; keep only explicit synthetic artifacts, never the database."""
    for name in _OUTPUTS:
        destination = output_dir / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="scholar-chunk-demo-") as temporary:
        database_path = Path(temporary) / "corpus.sqlite3"
        settings = offline_settings(database_path)
        app = create_app(settings)
        with TestClient(app) as client:
            ingested = client.post(
                "/ingest/text",
                json={
                    "title": "Synthetic methods and limitations",
                    "source": "synthetic:chunk-reader:selected",
                    "text": _TEXT,
                },
            )
            ingested.raise_for_status()
            excluded = client.post(
                "/ingest/text",
                json={
                    "title": "Synthetic excluded note",
                    "source": "synthetic:chunk-reader:excluded",
                    "text": "EXCLUDED_EVIDENCE should not be in selected pages.",
                },
            )
            excluded.raise_for_status()
            app.state.container.document_store.add_documents(
                [
                    Document(
                        document_id="imported-long",
                        title="Synthetic oversized imported chunk",
                        text="This intentionally bypasses the normal 800-character chunker.",
                        source="synthetic:chunk-reader:imported",
                    )
                ],
                [
                    Chunk(
                        chunk_id="imported-chunk",
                        document_id="imported-long",
                        title="Synthetic oversized imported chunk",
                        text="Synthetic long imported evidence. " * 200,
                        source="synthetic:chunk-reader:imported",
                        metadata={"chunk_index": "7", "private": "UNEXPOSED_METADATA"},
                    )
                ],
            )

        app = create_app(settings)
        container: AppContainer = app.state.container
        before = database_path.read_bytes()
        events_before = container.event_log.list_events()
        with TestClient(app) as client, ExitStack() as guards:
            for target, method in (
                (container.document_store, "list_chunks"),
                (container.document_store, "add_documents"),
                (container.hybrid_retriever, "retrieve"),
                (container.hybrid_retriever, "add_chunks"),
                (container.llm, "generate"),
                (container.runner, "run"),
                (container.event_log, "append_event"),
            ):
                guards.enter_context(patch.object(target, method, side_effect=_no_work))
            catalog_response = client.get(
                "/documents", params={"source": "synthetic:chunk-reader:selected"}
            )
            catalog_response.raise_for_status()
            catalog = DocumentCatalogPage.model_validate_json(catalog_response.content)
            paper = catalog.documents[0]
            endpoint = f"/documents/{quote(paper.document_id, safe='')}/chunks"
            pages = []
            responses = []
            cursor = None
            for _ in range(3):
                params: dict[str, str | int] = {"limit": 2}
                if cursor is not None:
                    params["cursor"] = cursor
                response = client.get(endpoint, params=params)
                response.raise_for_status()
                page = DocumentChunksPage.model_validate_json(response.content)
                responses.append(response)
                pages.append(page)
                cursor = page.next_cursor
            chunks = [chunk for page in pages for chunk in page.chunks]
            if (
                len(chunks) != paper.chunk_count
                or len({chunk.chunk_id for chunk in chunks}) != paper.chunk_count
                or [chunk.chunk_id for chunk in chunks]
                != sorted(chunk.chunk_id for chunk in chunks)
                or {chunk.document_id for chunk in chunks} != {paper.document_id}
                or any("EXCLUDED_EVIDENCE" in chunk.text for chunk in chunks)
                or pages[0].next_cursor is None
                or pages[1].next_cursor is None
                or pages[2].next_cursor is not None
            ):
                raise RuntimeError("The three evidence pages lost, repeated, or mis-scoped chunks.")
            long_response = client.get("/documents/imported-long/chunks")
            long_response.raise_for_status()
            long_page = DocumentChunksPage.model_validate_json(long_response.content)
            long_chunk = long_page.chunks[0]
            if (
                len(long_chunk.text) != 4000
                or not long_chunk.text_truncated
                or long_chunk.chunk_index != 7
                or "UNEXPOSED_METADATA" in long_response.text
                or any(response.headers["cache-control"] != "no-store" for response in responses)
            ):
                raise RuntimeError("The bounded imported-evidence contract was not preserved.")
        events_after = container.event_log.list_events()
        if database_path.read_bytes() != before or events_after != events_before:
            raise RuntimeError("Read-only evidence inspection changed the stored corpus or events.")

    artifacts = {
        "catalog.json": catalog_response.content,
        **{f"page-{index}.json": response.content for index, response in enumerate(responses, 1)},
        "long-chunk.json": long_response.content,
        "query-request.json": json.dumps(
            {
                "query": "Compare the synthetic methods and limitations",
                "document_ids": [paper.document_id],
            },
            indent=2,
        ).encode("utf-8"),
    }
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)
    transcript = (
        "\n\n".join(
            (
                "\n".join(
                    (
                        PANEL_TITLES[0],
                        f"POST /ingest/text -> {ingested.status_code}; synthetic selected note",
                        f"Stored selected chunks: {len(ingested.json()['chunk_ids'])}",
                        "Also saved an excluded note and an oversized imported fixture.",
                        "Recreated the application using the same temporary SQLite database.",
                        "Settings ignore environment, dotenv, and file secrets; no model calls.",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[1],
                        "GET /documents?source=<selected-source> -> "
                        f"{catalog_response.status_code}",
                        f"Recovered ID: {paper.document_id}",
                        f"GET /documents/<ID>/chunks?limit=2 -> {responses[0].status_code}",
                        f"Page 1: {len(pages[0].chunks)} chunks; next_cursor present",
                        f"First stored ordinal: {pages[0].chunks[0].chunk_index}",
                        f"Text: {pages[0].chunks[0].text[:76]}",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[2],
                        "Continue with the exact returned cursor for the same document.",
                        f"Page sizes: {', '.join(str(len(page.chunks)) for page in pages)}",
                        f"Distinct chunks: {len(chunks)}; "
                        f"final next_cursor={pages[-1].next_cursor}",
                        "Order: ascending opaque chunk IDs, not paragraph or relevance order.",
                        "Excluded-document chunks returned: 0",
                        f"Events before/after browsing: {len(events_before)}/{len(events_after)}",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[3],
                        f"Imported chunk -> {long_response.status_code}; "
                        f"text characters={len(long_chunk.text)}",
                        f"text_truncated={long_chunk.text_truncated}; "
                        f"stored chunk_index={long_chunk.chunk_index}",
                        "SQLite bytes unchanged by browsing; temporary database removed.",
                        "Saved query-request.json with the discovered document_ids.",
                        "Query is a saved request only: no generation or retrieval was run.",
                        "Synthetic evidence is not scientific verification or a research UI.",
                    )
                ),
            )
        )
        + "\n"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def main() -> None:
    """Run a reproducible synthetic workflow with no server or external corpus downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
