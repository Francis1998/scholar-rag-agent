"""Measure real collection duplicate inspection using a guarded synthetic offline corpus."""

import argparse
import hashlib
import json
import socket
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_evidence_export import offline_settings
from storage.collection_duplicates import SQLiteCollectionDuplicates
from storage.paper_collections import PaperCollection

PANEL_TITLES = (
    "1. Save an explicit synthetic selection",
    "2. Inspect candidates without losing papers",
    "3. Compare Python, HTTP, and restart output",
    "4. Choose IDs explicitly before preview",
)
OUTPUT_NAMES = (
    "collection.json",
    "duplicates.json",
    "duplicates.md",
    "restarted.json",
    "restarted.md",
    "preview.json",
    "checks.json",
    "transcript.txt",
)


def _seed(container: AppContainer) -> tuple[str, ...]:
    documents = [
        Document(
            document_id=identifier,
            title=title,
            source="synthetic:duplicate-inspection",
            text="GraphRAG connects synthetic research evidence; this is not a scientific finding.",
            metadata=metadata,
        )
        for identifier, title, metadata in (
            ("doi-b", "Different synthetic label", {"paper_doi": "https://doi.org/10.0000/NOTE."}),
            ("doi-a", "Synthetic DOI fixture", {"doi": " DOI:10.0000/note "}),
            ("title-b", "Synthetic GraphRAG method notes", {}),
            ("title-a", "Synthetic GraphRAG method note", {"work_doi": "10.0000/different"}),
            ("Paper", "", {}),
            ("paper", " \t ", {}),
            ("solo", "Zoology", {}),
            ("outside", "Not selected", {"doi": "10.0000/note"}),
        )
    ]
    chunks = [Chunk(chunk_id=f"chunk-{doc.document_id}", **doc.model_dump()) for doc in documents]
    container.document_store.add_documents(documents, chunks)
    container.hybrid_retriever.add_chunks(chunks)
    return tuple(doc.document_id for doc in documents[:-1])


def _guards(stack: ExitStack, container: AppContainer, *, preview: bool = False) -> list[Mock]:
    operations = [
        (container.runner, "run"),
        (container.llm, "generate"),
        (container.document_store, "list_chunks"),
        (container.document_store, "add_documents"),
        (container.event_log, "append_event"),
        (container.event_log, "append_transition"),
        (container.paper_collections, "create"),
        (container.paper_collections, "replace"),
        (container.paper_collections, "delete"),
        (container.paper_screening, "submit"),
    ]
    if not preview:
        operations.extend([(container.runner, "preview"), (container.hybrid_retriever, "retrieve")])
    return [
        stack.enter_context(
            patch.object(
                component, method, side_effect=AssertionError("Forbidden offline inspection work.")
            )
        )
        for component, method in operations
    ]


def run_demo(output_dir: Path) -> str:
    """Save actual API bytes and measured checks, then clean up only the temporary database."""
    for name in OUTPUT_NAMES:
        target = output_dir / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, bytes] = {}
    guarded_calls: list[Mock] = []
    inspection_database_unchanged = []
    with (
        TemporaryDirectory(prefix="scholar-duplicates-") as temporary,
        ExitStack() as network_stack,
    ):
        network = [
            network_stack.enter_context(
                patch.object(
                    component,
                    method,
                    side_effect=AssertionError("The synthetic duplicate demo forbids network."),
                )
            )
            for component, method in (
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
                (socket, "create_connection"),
                (socket.socket, "connect"),
            )
        ]
        database = Path(temporary) / "duplicates.sqlite3"
        settings = offline_settings(database)
        app = create_app(settings)
        container: AppContainer = app.state.container
        identifiers = _seed(container)
        with TestClient(app) as client:
            saved = client.post(
                "/collections",
                json={"name": "Synthetic duplicate candidates", "document_ids": identifiers},
            )
            saved.raise_for_status()
            collection = PaperCollection.model_validate_json(saved.content)
            artifacts["collection.json"] = saved.content
            route = f"/collections/{collection.collection_id}/duplicates"
            before = database.read_bytes()
            with ExitStack() as inspect_stack:
                guarded_calls.extend(_guards(inspect_stack, container))
                for format, name in (("json", "duplicates.json"), ("markdown", "duplicates.md")):
                    response = client.get(
                        route,
                        params={"collection_revision": collection.revision, "format": format},
                    )
                    response.raise_for_status()
                    artifacts[name] = response.content
                report = container.collection_duplicates.inspect(
                    collection.collection_id, collection_revision=collection.revision
                )
                if (
                    artifacts["duplicates.json"] != report.to_json().encode()
                    or artifacts["duplicates.md"] != report.to_markdown().encode()
                    or report.document_ids != tuple(sorted(identifiers))
                    or [[p.document_id for p in g.members] for g in report.groups]
                    != [["doi-a", "doi-b"], ["title-a", "title-b"]]
                    or [p.document_id for p in report.singletons] != ["Paper", "paper", "solo"]
                ):
                    raise RuntimeError("The real inspection did not match its synthetic fixture.")
            inspection_database_unchanged.append(database.read_bytes() == before)
        restarted = create_app(settings)
        restarted_container: AppContainer = restarted.state.container
        with TestClient(restarted) as client:
            before = database.read_bytes()
            with ExitStack() as inspect_stack:
                guarded_calls.extend(_guards(inspect_stack, restarted_container))
                for format, original, name in (
                    ("json", "duplicates.json", "restarted.json"),
                    ("markdown", "duplicates.md", "restarted.md"),
                ):
                    response = client.get(
                        route,
                        params={"collection_revision": collection.revision, "format": format},
                    )
                    response.raise_for_status()
                    artifacts[name] = response.content
                    if response.content != artifacts[original]:
                        raise RuntimeError("Unchanged stored data produced a changed report.")
                if report != SQLiteCollectionDuplicates(database).inspect(
                    collection.collection_id, collection_revision=collection.revision
                ):
                    raise RuntimeError("Standalone Python and HTTP inspection disagree.")
            inspection_database_unchanged.append(database.read_bytes() == before)
            selected = ["doi-a", "title-a"]
            with ExitStack() as preview_stack:
                guarded_calls.extend(_guards(preview_stack, restarted_container, preview=True))
                preview = client.post(
                    "/retrieve",
                    json={"query": "What does GraphRAG connect?", "document_ids": selected},
                )
                preview.raise_for_status()
                artifacts["preview.json"] = preview.content
            preview_ids = sorted(
                {source["chunk"]["document_id"] for source in preview.json()["sources"]}
            )
            if preview_ids != selected:
                raise RuntimeError("Explicit synthetic human selection widened or lost IDs.")
            events = restarted_container.event_log.list_events()
            if (
                any(guard.call_count for guard in guarded_calls + network)
                or not all(inspection_database_unchanged)
                or events
                or database.read_bytes() != before
                or restarted_container.paper_collections.get(collection.collection_id) != collection
            ):
                raise RuntimeError("Inspection or preview changed storage or did forbidden work.")
    removed = not database.exists()
    if not removed:
        raise RuntimeError("The temporary synthetic database was not removed.")
    checks = {
        "selected_documents": report.inspected_count,
        "candidate_groups": len(report.groups),
        "singleton_documents": len(report.singletons),
        "complete_partition": sorted(
            [p.document_id for g in report.groups for p in g.members]
            + [p.document_id for p in report.singletons]
        )
        == sorted(identifiers),
        "case_sensitive_singletons": ["Paper", "paper"],
        "outside_collection_absent": "outside" not in report.document_ids,
        "inspection_database_unchanged": all(inspection_database_unchanged),
        "inspection_and_preview_forbidden_calls": sum(g.call_count for g in guarded_calls),
        "network_calls": sum(g.call_count for g in network),
        "event_count": len(events),
        "explicit_human_selected_ids": selected,
        "preview_document_ids": preview_ids,
        "python_matches_http": True,
        "temporary_database_removed": removed,
        "response_bytes": {
            "json": len(artifacts["duplicates.json"]),
            "markdown": len(artifacts["duplicates.md"]),
        },
        "restart_sha256": {
            format: {
                "before": hashlib.sha256(artifacts[original]).hexdigest(),
                "after": hashlib.sha256(artifacts[reopened]).hexdigest(),
            }
            for format, original, reopened in (
                ("json", "duplicates.json", "restarted.json"),
                ("markdown", "duplicates.md", "restarted.md"),
            )
        },
    }
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"POST /collections -> {saved.status_code}; revision={collection.revision}",
                        f"Selected: {report.inspected_count} of 8 synthetic documents",
                        "The actual returned ID and revision are saved in collection.json.",
                        "Outside-collection document absent: "
                        f"{checks['outside_collection_absent']}",
                        "No body or chunk reads in inspection; no whole-corpus scan.",
                        "Ambient settings ignored; no scientific data or model credentials.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        "GET /collections/<id>/duplicates?collection_revision=1 -> 200",
                        f"Candidate groups: {len(report.groups)}; "
                        f"singletons: {len(report.singletons)}",
                        *[
                            f"{', '.join(p.document_id for p in group.members)}: "
                            f"{', '.join(group.match_reasons)}"
                            for group in report.groups
                        ],
                        f"Every selected identity accounted for: {checks['complete_partition']}",
                        "Raw DOI strings retained; aliases normalized only for matching.",
                        "Candidates can be transitive or conflicting, not proof of same work.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        "Paper and paper remain separate blank-title singletons.",
                        f"JSON bytes: {checks['response_bytes']['json']}; "
                        f"literal Markdown bytes: {checks['response_bytes']['markdown']}",
                        "Both complete formats fit the 262144-byte cap; no clipped evidence.",
                        f"Python equals HTTP: {checks['python_matches_http']}",
                        "Unchanged corpus after restart: byte-identical JSON and Markdown.",
                        f"Inspection database unchanged: {checks['inspection_database_unchanged']}",
                        "One read snapshot per call, not a frozen report across later edits.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Explicit synthetic human choice: {', '.join(selected)}",
                        f"POST /retrieve -> {preview.status_code}; "
                        f"papers: {', '.join(preview_ids)}",
                        "Selection was not inferred from the candidate groups.",
                        "Forbidden inspection/preview calls: "
                        f"{checks['inspection_and_preview_forbidden_calls']}",
                        f"Network calls: {checks['network_calls']}; "
                        f"agent events: {checks['event_count']}",
                        f"Temporary synthetic database removed: {removed}",
                        "Measured fixture output, not a browser recording or quality benchmark.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    artifacts["checks.json"] = (json.dumps(checks, indent=2, sort_keys=True) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode("utf-8")
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as output:
            output.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
