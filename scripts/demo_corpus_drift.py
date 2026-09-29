"""Measure frozen-evidence drift using synthetic data, real API routes, and no live HTTP."""

import argparse
import json
import socket
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle, text_digest
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from retrieval.models import Chunk, Document
from scripts.demo_evidence_export import offline_settings
from storage.corpus_drift import CorpusDriftReport

PANEL_TITLES = (
    "1. Save five synthetic evidence chunks",
    "2. Inspect persisted chunks without generation",
    "3. Explain exact changes, not scientific validity",
    "4. Restart without rewriting frozen evidence",
)
OUTPUT_NAMES = (
    "bundle.json",
    "bundle.md",
    "unchanged.json",
    "drift.json",
    "restarted.json",
    "checks.json",
    "transcript.txt",
)


def _seed(container: AppContainer) -> list[Chunk]:
    chunks = [
        Chunk(
            chunk_id=f"demo-{label}",
            document_id=f"paper-{label}",
            title="Synthetic GraphRAG evidence",
            text=f"GraphRAG connects synthetic research evidence for {label}. Not a real finding.",
            source="synthetic:corpus-drift",
            metadata={"fixture": "synthetic-only", "revision": "1"},
        )
        for label in ("unchanged", "changed", "missing-chunk", "missing-document", "reassigned")
    ]
    container.document_store.add_documents(
        [Document(**chunk.model_dump(exclude={"chunk_id"})) for chunk in chunks], chunks
    )
    container.hybrid_retriever.add_chunks(chunks)
    container.graph_builder.index_chunks(chunks)
    return chunks


def _block_work(stack: ExitStack, container: AppContainer) -> list[Mock]:
    return [
        stack.enter_context(
            patch.object(
                component, method, side_effect=AssertionError("Drift must not perform agent work.")
            )
        )
        for component, method in (
            (container.runner, "run"),
            (container.runner, "preview"),
            (container.llm, "generate"),
            (container.hybrid_retriever, "retrieve"),
            (container.document_store, "list_chunks"),
            (container.graph_store, "chunks_for_entities"),
            (container.event_log, "append_event"),
            (container.event_log, "append_transition"),
        )
    ]


def run_demo(output_dir: Path) -> str:
    """Leave measured artifacts, not a database; ignore ambient keys/settings and block HTTP."""
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, bytes] = {}
    with TemporaryDirectory(prefix="scholar-corpus-drift-") as temporary, ExitStack() as stack:
        network = [
            stack.enter_context(
                patch.object(
                    transport,
                    method,
                    side_effect=AssertionError("The synthetic demo forbids network access."),
                )
            )
            for transport, method in (
                (httpx.HTTPTransport, "handle_request"),
                (httpx.AsyncHTTPTransport, "handle_async_request"),
                (socket, "create_connection"),
                (socket.socket, "connect"),
            )
        ]
        database_path = Path(temporary) / "drift.sqlite3"
        settings = offline_settings(database_path)
        application = create_app(settings)
        container: AppContainer = application.state.container
        chunks = _seed(container)
        with TestClient(application) as client:
            query = client.post("/query", json={"query": "What does GraphRAG evidence connect?"})
            query.raise_for_status()
            result = QueryResponse.model_validate_json(query.content).result
            if result.state != "DONE":
                raise RuntimeError(f"The synthetic query failed: {result.error}")
            export_path = f"/runs/{result.run_id}/export"
            drift_path = f"/runs/{result.run_id}/corpus-drift"
            for fmt, name in (("json", "bundle.json"), ("markdown", "bundle.md")):
                response = client.get(export_path, params={"format": fmt})
                response.raise_for_status()
                artifacts[name] = response.content
            bundle = EvidenceBundle.model_validate_json(artifacts["bundle.json"])
            if bundle.generation.provider != "fake" or len(bundle.snapshot.sources) != 5:
                raise RuntimeError("Expected a fake-adapter run with five synthetic passages.")
            before_events = container.event_log.list_events()
            blocked = _block_work(stack, container)
            unchanged = client.get(drift_path)
            unchanged.raise_for_status()
            initial = CorpusDriftReport.model_validate_json(unchanged.content)
            if initial.counts.unchanged != 5 or initial.has_drift:
                raise RuntimeError("The unmodified synthetic corpus did not match.")
            artifacts["unchanged.json"] = unchanged.content

            replacement = chunks[1].model_copy(
                update={
                    "text": "Updated synthetic evidence: caf\u00e9, \u7814\u7a76.",
                    "title": "Re-ingested synthetic title",
                    "source": "synthetic:replacement",
                    "metadata": {"fixture": "synthetic-only", "revision": "2"},
                }
            )
            container.document_store.add_documents(
                [Document(**replacement.model_dump(exclude={"chunk_id"}))], [replacement]
            )
            # Only this newly created temporary synthetic corpus is changed.
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute("DELETE FROM chunks WHERE chunk_id = 'demo-missing-chunk'")
                connection.execute(
                    "DELETE FROM documents WHERE document_id = 'paper-missing-document'"
                )
                connection.execute(
                    "UPDATE chunks SET document_id = 'another-owner' "
                    "WHERE chunk_id = 'demo-reassigned'"
                )
            drift = client.get(drift_path)
            drift.raise_for_status()
            report = CorpusDriftReport.model_validate_json(drift.content)
            by_id = {source.chunk_id: source for source in report.sources}
            expected_counts = {"total": 5, "unchanged": 1, "changed": 1, "missing": 3}
            expected_fields = ["text", "title", "source", "metadata"]
            if (
                report.counts.model_dump() != expected_counts
                or by_id["demo-changed"].changed_fields != expected_fields
                or by_id["demo-missing-chunk"].missing_reason != "chunk_missing"
                or by_id["demo-missing-document"].missing_reason != "document_missing"
                or by_id["demo-reassigned"].missing_reason != "chunk_reassigned"
                or [source.chunk_id for source in report.sources]
                != [source.chunk.chunk_id for source in bundle.snapshot.sources]
            ):
                raise RuntimeError("The report did not match the actual persisted mutations.")
            artifacts["drift.json"] = drift.content

        reopened = create_app(settings)
        blocked.extend(_block_work(stack, reopened.state.container))
        with TestClient(reopened) as client:
            restarted = client.get(drift_path)
            restarted.raise_for_status()
            artifacts["restarted.json"] = restarted.content
            if restarted.content != drift.content:
                raise RuntimeError("The persisted drift report changed after restart.")
            digests = {}
            for fmt, name in (("json", "bundle.json"), ("markdown", "bundle.md")):
                after = client.get(export_path, params={"format": fmt})
                after.raise_for_status()
                if after.content != artifacts[name]:
                    raise RuntimeError(f"The frozen {fmt} export was rewritten.")
                digests[fmt] = {
                    "before": text_digest(artifacts[name].decode("utf-8")),
                    "after": text_digest(after.text),
                }
            after_events = reopened.state.container.event_log.list_events()
        if before_events != after_events:
            raise RuntimeError("The report changed saved run events.")
        work_calls = sum(guard.call_count for guard in blocked)
        network_calls = sum(guard.call_count for guard in network)
        event_writes = len(after_events) - len(before_events)
        if work_calls or network_calls or event_writes:
            raise RuntimeError("Drift inspection performed unexpected work.")
    database_removed = not database_path.exists()
    if not database_removed:
        raise RuntimeError("The temporary synthetic database was not removed.")
    checks = {
        "export_sha256": digests,
        "report_restart_byte_equal": drift.content == restarted.content,
        "events_unchanged": before_events == after_events,
        "drift_agent_calls": work_calls,
        "drift_event_writes": event_writes,
        "network_calls": network_calls,
        "temporary_database_removed": database_removed,
    }
    artifacts["checks.json"] = (json.dumps(checks, indent=2, sort_keys=True) + "\n").encode()
    counts = report.counts
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"POST /query -> {query.status_code}; state={result.state}",
                        f"Saved provider={bundle.generation.provider}; chunks={len(chunks)}",
                        "Synthetic notes and a fake answer, not scientific findings.",
                        "Frozen exports: bundle.json and bundle.md",
                        f"Live HTTP calls: {network_calls}; ambient settings ignored",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"GET /runs/<run_id>/corpus-drift -> {unchanged.status_code}",
                        f"Before edits: unchanged={initial.counts.unchanged}",
                        "Re-ingest the same IDs, then remove and reassign synthetic chunks.",
                        f"After edits: unchanged={counts.unchanged}, changed={counts.changed}, "
                        f"missing={counts.missing}",
                        f"Drift agent calls: {work_calls}; event writes: {event_writes}",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        "Changed fields: " + ", ".join(by_id["demo-changed"].changed_fields),
                        "Missing chunk: " + str(by_id["demo-missing-chunk"].missing_reason),
                        "Missing document: " + str(by_id["demo-missing-document"].missing_reason),
                        "Reassigned ID: " + str(by_id["demo-reassigned"].missing_reason),
                        "Frozen source order retained; current text and metadata not returned.",
                        "Unchanged chunks do not imply whole-paper equality or valid science.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        "Report after restart: byte-identical",
                        "Frozen JSON and Markdown exports: byte-identical",
                        f"Saved run events unchanged: {checks['events_unchanged']}",
                        f"Temporary demo database removed: {database_removed}",
                        "Artifact checks and digests: checks.json",
                        "Measured offline output, not a live research UI or new model answer.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    artifacts["transcript.txt"] = transcript.encode("utf-8")
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
