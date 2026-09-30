"""Measure cited-only bibliography exports using synthetic saved evidence and no live HTTP."""

import argparse
import hashlib
import json
import socket
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from retrieval.models import Chunk, Document
from scripts.demo_evidence_export import offline_settings
from storage.saved_bibliography import SavedBibliography

PANEL_TITLES = (
    "1. Save synthetic evidence and a fake answer",
    "2. Download actual BibTeX, not a new answer",
    "3. Keep cited identities and disclose conflicts",
    "4. Remove the demo corpus, restart, repeat",
)
OUTPUT_NAMES = (
    "evidence.json",
    "evidence.md",
    "bibliography.json",
    "bibliography.bib",
    "restarted.json",
    "restarted.bib",
    "checks.json",
    "transcript.txt",
)


def _seed(container: AppContainer) -> None:
    text = (
        "GraphRAG connects synthetic research evidence. "
        "This is a synthetic workshop note, not a publication or scientific finding."
    )
    chunks = [
        Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title=title,
            text=text,
            source="synthetic:saved-bibliography",
            metadata={"author": author, "year": year, "fixture": "synthetic-only"},
        )
        for chunk_id, document_id, title, author, year in (
            ("a-1", "paper-a", "Synthetic reference A", "Ada Example", "2024"),
            ("a-2", "paper-a", "Synthetic reference A", "Ada Example", "2025"),
            ("b-1", "paper-b", "Synthetic reference B", "Bea Example", "2023"),
            ("uncited-1", "uncited-paper", "Retrieved but uncited note", "Not imported", "2022"),
        )
    ]
    container.document_store.add_documents(
        [
            Document(**chunk.model_dump(exclude={"chunk_id"}))
            for chunk in (chunks[0], chunks[2], chunks[3])
        ],
        chunks,
    )
    container.hybrid_retriever.add_chunks(chunks)
    container.graph_builder.index_chunks(chunks)


def _block_export_work(stack: ExitStack, container: AppContainer) -> list[Mock]:
    return [
        stack.enter_context(
            patch.object(
                component,
                method,
                side_effect=AssertionError("A bibliography export attempted agent/corpus work."),
            )
        )
        for component, method in (
            (container.runner, "run"),
            (container.runner, "preview"),
            (container.llm, "generate"),
            (container.hybrid_retriever, "retrieve"),
            (container.document_store, "list_chunks"),
            (container.document_store, "add_documents"),
            (container.graph_store, "chunks_for_entities"),
            (container.event_log, "append_event"),
            (container.event_log, "append_transition"),
        )
    ]


def run_demo(output_dir: Path) -> str:
    """Save actual API bytes and measured checks; remove only the new synthetic database."""
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, bytes] = {}
    with TemporaryDirectory(prefix="scholar-bibliography-") as temporary, ExitStack() as stack:
        network = [
            stack.enter_context(
                patch.object(
                    target,
                    method,
                    side_effect=AssertionError("The synthetic bibliography demo forbids network."),
                )
            )
            for target, method in (
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
                (socket, "create_connection"),
                (socket.socket, "connect"),
            )
        ]
        database_path = Path(temporary) / "bibliography.sqlite3"
        settings = offline_settings(database_path)
        application = create_app(settings)
        container: AppContainer = application.state.container
        _seed(container)
        with TestClient(application) as client:
            query = client.post("/query", json={"query": "What does GraphRAG connect?"})
            query.raise_for_status()
            result = QueryResponse.model_validate_json(query.content).result
            if result.state != "DONE":
                raise RuntimeError(f"The synthetic query failed: {result.error}")
            export_path = f"/runs/{result.run_id}/export"
            bibliography_path = f"/runs/{result.run_id}/bibliography"
            for format, name in (("json", "evidence.json"), ("markdown", "evidence.md")):
                response = client.get(export_path, params={"format": format})
                response.raise_for_status()
                artifacts[name] = response.content
            bundle = EvidenceBundle.model_validate_json(artifacts["evidence.json"])
            if bundle.generation.provider != "fake" or len(bundle.snapshot.sources) != 4:
                raise RuntimeError("Expected one fake-adapter query with four synthetic chunks.")
            events_before = container.event_log.list_events()
            blocked = _block_export_work(stack, container)
            bytes_before = database_path.read_bytes()
            json_response = client.get(bibliography_path, params={"format": "json"})
            bibtex_response = client.get(bibliography_path)
            json_response.raise_for_status()
            bibtex_response.raise_for_status()
            report = SavedBibliography.model_validate_json(json_response.content)
            if (
                len(bundle.answer.citations) != 3
                or [source.document_id for source in report.sources] != ["paper-a", "paper-b"]
                or bibtex_response.content != report.bibtex.encode("utf-8")
                or "uncited-paper" in report.to_json()
                or report.bibtex.count("@misc{") != 2
                or "doi =" in report.bibtex
            ):
                raise RuntimeError("The actual bibliography did not match its cited-only fixture.")
            conflicts = [
                warning for warning in report.warnings if warning.startswith("Conflicting")
            ]
            if len(conflicts) != 1 or report.sources[0].metadata["year"] != "2024":
                raise RuntimeError("Conflicting years were not disclosed with first-rank choice.")
            artifacts["bibliography.json"] = json_response.content
            artifacts["bibliography.bib"] = bibtex_response.content
            if database_path.read_bytes() != bytes_before:
                raise RuntimeError("The bibliography request wrote to the database.")
            # Only this freshly created synthetic database is changed, outside export requests.
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute("UPDATE chunks SET title = 'Replaced synthetic metadata'")
                connection.execute("DELETE FROM chunks")
                connection.execute("DELETE FROM documents")
                remaining_chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        restarted = create_app(settings)
        blocked.extend(_block_export_work(stack, restarted.state.container))
        bytes_before = database_path.read_bytes()
        with TestClient(restarted) as client:
            for format, original, name in (
                ("json", "bibliography.json", "restarted.json"),
                ("bibtex", "bibliography.bib", "restarted.bib"),
            ):
                after = client.get(bibliography_path, params={"format": format})
                after.raise_for_status()
                artifacts[name] = after.content
                if after.content != artifacts[original]:
                    raise RuntimeError(f"The {format} bibliography changed after corpus removal.")
            for format, name in (("json", "evidence.json"), ("markdown", "evidence.md")):
                after = client.get(export_path, params={"format": format})
                after.raise_for_status()
                if after.content != artifacts[name]:
                    raise RuntimeError("The original frozen evidence export changed.")
        events_after = restarted.state.container.event_log.list_events()
        work_calls = sum(guard.call_count for guard in blocked)
        network_calls = sum(guard.call_count for guard in network)
        if (
            work_calls
            or network_calls
            or remaining_chunks != 0
            or events_before != events_after
            or database_path.read_bytes() != bytes_before
        ):
            raise RuntimeError("Export performed unexpected work or changed persisted state.")
    database_removed = not database_path.exists()
    if not database_removed:
        raise RuntimeError("The temporary synthetic database was not removed.")
    checks = {
        "captured_sources": len(bundle.snapshot.sources),
        "final_citations": len(bundle.answer.citations),
        "cited_documents": len(report.sources),
        "metadata_conflicts": len(conflicts),
        "remaining_corpus_chunks": remaining_chunks,
        "bibliography_bytes": {
            "json": len(json_response.content),
            "bibtex": len(bibtex_response.content),
        },
        "bibliography_sha256": {
            format: {
                "before": hashlib.sha256(artifacts[original]).hexdigest(),
                "after": hashlib.sha256(artifacts[reopened]).hexdigest(),
            }
            for format, original, reopened in (
                ("json", "bibliography.json", "restarted.json"),
                ("bibtex", "bibliography.bib", "restarted.bib"),
            )
        },
        "evidence_sha256": {
            name: hashlib.sha256(artifacts[name]).hexdigest()
            for name in ("evidence.json", "evidence.md")
        },
        "events_unchanged": events_before == events_after,
        "export_agent_or_corpus_calls": work_calls,
        "export_event_writes": len(events_after) - len(events_before),
        "network_calls": network_calls,
        "temporary_database_removed": database_removed,
    }
    artifacts["checks.json"] = (json.dumps(checks, indent=2, sort_keys=True) + "\n").encode()
    mapping = [
        f"{chunk.chunk_id} -> {source.document_id}; frozen rank {chunk.evidence_rank}; "
        f"citation positions {chunk.citation_numbers}"
        for source in report.sources
        for chunk in source.cited_chunks
    ]
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"POST /query -> {query.status_code}; state={result.state}",
                        f"Saved provider: {bundle.generation.provider} "
                        "(placeholder, not scientific output)",
                        f"Frozen chunks: {len(bundle.snapshot.sources)}; final citations: "
                        f"{len(bundle.answer.citations)}",
                        f"Cited documents: {len(report.sources)}; uncited chunk excluded",
                        f"Live HTTP/socket calls: {network_calls}; ambient settings ignored",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"GET /runs/<run_id>/bibliography -> {bibtex_response.status_code}",
                        "Excerpt from the actual downloaded bibliography.bib:",
                        report.bibtex.split("\n\n")[0],
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        *mapping,
                        f"Metadata conflicts disclosed: {len(conflicts)}; first cited rank wins",
                        f"Chosen paper-a year: {report.sources[0].metadata['year']}; "
                        "no field merging",
                        "JSON keeps exact document IDs, citation mapping, BibTeX, and warnings.",
                        "No DOI lookup. Missing metadata stays missing; review before importing.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Remaining corpus chunks: {remaining_chunks}",
                        "BibTeX and JSON after restart: byte-identical",
                        "Original frozen JSON and Markdown evidence: byte-identical",
                        f"Export agent/corpus calls: {work_calls}; "
                        f"event writes: {checks['export_event_writes']}",
                        f"Saved events unchanged: {checks['events_unchanged']}",
                        f"Temporary database removed: {database_removed}; digests in checks.json",
                        "Actual offline output, not a research UI or reference-quality benchmark.",
                    ]
                ),
            ]
        )
        + "\n"
    )
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
