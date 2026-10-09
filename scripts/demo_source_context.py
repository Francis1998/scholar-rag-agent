"""Measure source-context reads using an isolated, temporary synthetic corpus."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

import httpx
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_corpus_explorer import ExplorerPage, browsing_guards
from scripts.demo_corpus_explorer import _no_work as _no_work
from scripts.demo_evidence_export import offline_settings
from storage.document_chunks import DocumentChunksPage
from storage.source_context import SourceContext, SQLiteSourceContext

SELECTED_ID = "paper/../\u7814?draft#v1"
SOURCE = "synthetic:source-context"
PANEL_TITLES = (
    "1. Discover a displayed passage",
    "2. Read the source-order window",
    "3. Recenter or keep the anchor",
    "4. Restart without changing evidence",
)
_OUTPUTS = (
    "catalog.json",
    "passages.json",
    "context.json",
    "recentered.json",
    "anchor-only.json",
    "edge-context.json",
    "passages.html",
    "context.html",
    "checks.json",
    "transcript.txt",
)


def seed_demo(container: AppContainer) -> None:
    """Create original imported fixtures, not scientific findings or downloaded papers."""
    document = Document(
        document_id=SELECTED_ID,
        title="Synthetic methods, observation and limitations",
        source=SOURCE,
        text="UNEXPOSED_DOCUMENT_BODY",
        metadata={"private": "UNEXPOSED_METADATA"},
    )
    passages = {
        8: "Synthetic methods. The fixture is a reading example, not a study.",
        9: "Synthetic preceding context. Assumptions belong beside the selected observation.",
        10: "Synthetic selected passage. A standalone observation is not proof of a conclusion.",
        11: "Synthetic limitation. Read the surrounding passages before interpreting this example.",
        12: "Synthetic long imported passage. Consult the permitted original for omitted text. "
        * 60,
    }
    chunks = [
        Chunk(
            chunk_id=f"chunk-{index}",
            document_id=document.document_id,
            title=document.title,
            source=document.source,
            text=passages.get(index, f"Synthetic stored passage {index}. No research findings."),
            metadata={"chunk_index": str(index), "private": "UNEXPOSED_METADATA"},
        )
        for index in range(14)
    ]
    excluded = Document(
        document_id="excluded",
        title="Another synthetic paper",
        source="synthetic:excluded",
        text="EXCLUDED_EVIDENCE",
    )
    chunks.append(
        Chunk(
            chunk_id="other-chunk",
            document_id=excluded.document_id,
            title=excluded.title,
            source=excluded.source,
            text=excluded.text,
        )
    )
    container.document_store.add_documents([document, excluded], chunks)


def _get(client: TestClient, path: str, **params: str | int) -> httpx.Response:
    response = client.get(path, params=params) if params else client.get(path)
    response.raise_for_status()
    if (
        response.headers["cache-control"] != "no-store"
        or response.headers["x-content-type-options"] != "nosniff"
        or "UNEXPOSED_" in response.text
        or "EXCLUDED_EVIDENCE" in response.text
    ):
        raise RuntimeError("A source-context read lost its headers or exact evidence scope.")
    return response


def run_demo(output_dir: Path) -> str:
    """Save real results and measurements; the temporary corpus is removed on exit."""
    if output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    for name in _OUTPUTS:
        destination = output_dir / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination}; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="scholar-source-context-") as temporary:
        path = Path(temporary) / "corpus.sqlite3"
        settings = offline_settings(path)
        app = create_app(settings)
        seed_demo(app.state.container)
        before = path.read_bytes()
        events_before = app.state.container.event_log.list_events()
        artifacts: dict[str, bytes] = {}
        with TestClient(app) as client, browsing_guards(app.state.container) as guards:
            catalog = _get(client, "/documents", source=SOURCE)
            paper = catalog.json()["documents"][0]
            passages = app.state.container.document_chunks.list_chunks(paper["document_id"])
            selected = next(chunk for chunk in passages.chunks if chunk.chunk_index == 10)
            document = _get(client, "/explore/document", document_id=paper["document_id"], limit=3)
            document_page = ExplorerPage(document.text)
            displayed_index = document_page.chunks.index(selected.chunk_id) + 1
            html = _get(client, document_page.links[f"source-context-{displayed_index}"])
            html_page = ExplorerPage(html.text)
            response = _get(
                client,
                "/documents/context",
                document_id=selected.document_id,
                chunk_id=selected.chunk_id,
            )
            result = SourceContext.model_validate_json(response.content)
            python_result = SQLiteSourceContext(path).read(selected.document_id, selected.chunk_id)
            recenter_url = html_page.links["center-context-4"]
            recentered_html = ExplorerPage(_get(client, recenter_url).text)
            recenter_id = parse_qs(urlsplit(recenter_url).query)["chunk_id"][0]
            recenter_response = _get(
                client, "/documents/context", document_id=selected.document_id, chunk_id=recenter_id
            )
            recentered = SourceContext.model_validate_json(recenter_response.content)
            anchor_response = _get(
                client,
                "/documents/context",
                document_id=selected.document_id,
                chunk_id=selected.chunk_id,
                before=0,
                after=0,
            )
            anchor_only = SourceContext.model_validate_json(anchor_response.content)
            edge_response = _get(
                client, "/documents/context", document_id=selected.document_id, chunk_id="chunk-13"
            )
            edge = SourceContext.model_validate_json(edge_response.content)
            missing = client.get(
                "/documents/context",
                params={"document_id": selected.document_id, "chunk_id": "missing-anchor"},
            )
            if (
                [chunk.chunk_index for chunk in result.chunks] != [8, 9, 10, 11, 12]
                or html_page.chunks != [chunk.chunk_id for chunk in result.chunks]
                or recentered_html.chunks != [chunk.chunk_id for chunk in recentered.chunks]
                or recentered.anchor_chunk_index != 11
                or [chunk.chunk_index for chunk in anchor_only.chunks] != [10]
                or [chunk.chunk_index for chunk in edge.chunks] != [11, 12, 13]
                or len(result.chunks[-1].text) != 4000
                or not result.chunks[-1].text_truncated
                or python_result != result
                or missing.status_code != 404
                or missing.json()["detail"]["code"] != "chunk_not_found"
            ):
                raise RuntimeError(
                    "Source context, native navigation, bounds or exact anchor failed."
                )
            artifacts.update(
                {
                    "catalog.json": catalog.content,
                    "passages.json": passages.model_dump_json().encode(),
                    "context.json": response.content,
                    "recentered.json": recenter_response.content,
                    "anchor-only.json": anchor_response.content,
                    "edge-context.json": edge_response.content,
                    "passages.html": document.content,
                    "context.html": html.content,
                }
            )
            counts = {name: guard.call_count for name, guard in guards.items()}
        restarted = create_app(settings)
        with TestClient(restarted) as client, browsing_guards(restarted.state.container) as guards:
            after_response = _get(
                client,
                "/documents/context",
                document_id=selected.document_id,
                chunk_id=selected.chunk_id,
            )
            after_html = _get(client, document_page.links[f"source-context-{displayed_index}"])
            restart_identical = (
                after_response.content == response.content and after_html.content == html.content
            )
            for name, guard in guards.items():
                counts[name] += guard.call_count
        events_after = restarted.state.container.event_log.list_events()
        unchanged = path.read_bytes() == before
        if (
            not restart_identical
            or not unchanged
            or events_after != events_before
            or any(counts.values())
        ):
            raise RuntimeError("Restart or reading changed evidence or attempted forbidden work.")
        checks = {
            "fixture": "scholar-source-context-synthetic-v1",
            "selected_document_id": selected.document_id,
            "anchor_chunk_id": result.anchor_chunk_id,
            "indices_in_id_order": [chunk.chunk_index for chunk in passages.chunks],
            "context_indices": [chunk.chunk_index for chunk in result.chunks],
            "recentered_indices": [chunk.chunk_index for chunk in recentered.chunks],
            "returned_before": result.returned_before,
            "returned_after": result.returned_after,
            "truncated_passage_characters": len(result.chunks[-1].text),
            "excluded_chunks_returned": sum(
                chunk.document_id != selected.document_id for chunk in result.chunks
            ),
            "python_api_identical": python_result == result,
            "restart_identical": restart_identical,
            "database_bytes_unchanged": unchanged,
            "events_before": len(events_before),
            "events_after": len(events_after),
            "missing_anchor_status": missing.status_code,
            "forbidden_call_counts": counts,
        }
    id_page = DocumentChunksPage.model_validate_json(artifacts["passages.json"])
    transcript = (
        "\n\n".join(
            (
                "\n".join(
                    (
                        PANEL_TITLES[0],
                        f"GET /documents -> {catalog.status_code}; "
                        f"matching papers={len(catalog.json()['documents'])}",
                        f"Stored passages in this paper: {len(id_page.chunks)}",
                        "ID-ordered indices: "
                        + ", ".join(str(chunk.chunk_index) for chunk in id_page.chunks),
                        f"Displayed anchor: {selected.chunk_id}; "
                        f"stored index={selected.chunk_index}",
                        "Original synthetic fixtures; isolated settings; no downloaded papers.",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[1],
                        f"GET /documents/context -> {response.status_code}",
                        "Source indices: "
                        + ", ".join(str(chunk.chunk_index) for chunk in result.chunks),
                        f"Returned {result.returned_before} before + selected passage + "
                        f"{result.returned_after} after.",
                        f"Native HTML link -> {html.status_code}; HTML order agrees with JSON.",
                        f"Excluded-document chunks returned: {checks['excluded_chunks_returned']}",
                        "Unique source indices, not chunk ID order or sentence parsing.",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[2],
                        f"Recenter on {recentered.anchor_chunk_id}: "
                        + ", ".join(str(chunk.chunk_index) for chunk in recentered.chunks),
                        f"Anchor-only window: {len(anchor_only.chunks)} passage; "
                        f"index={anchor_only.anchor_chunk_index}",
                        f"End window: {edge.returned_before} before + selected passage + "
                        f"{edge.returned_after} after.",
                        f"Long passage prefix: {len(result.chunks[-1].text)} characters; "
                        f"truncated={result.chunks[-1].text_truncated}",
                        f"Missing exact anchor -> {missing.status_code}; no guessed replacement.",
                        "Native links preserve the original passage-page return path.",
                    )
                ),
                "\n".join(
                    (
                        PANEL_TITLES[3],
                        f"Python/API identical: {checks['python_api_identical']}; "
                        f"restart identical: {restart_identical}",
                        f"SQLite bytes unchanged by reads/restart: {unchanged}",
                        f"Events before/after: {len(events_before)}/{len(events_after)}",
                        "Forbidden read-call attempts (model/retrieval/HTTP/writes): "
                        f"{sum(counts.values())}",
                        "CURRENT CORPUS, not frozen saved-run evidence or scientific validation.",
                        "JSON, HTML, checks and transcript saved; temporary database removed.",
                    )
                ),
            )
        )
        + "\n"
    )
    artifacts["checks.json"] = (json.dumps(checks, ensure_ascii=True, indent=2) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode()
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as destination:
            destination.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
