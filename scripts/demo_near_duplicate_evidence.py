"""Measure opt-in evidence collapse through real routes without any model or network call."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from agent.evidence import EvidenceSnapshot
from agent.models import AgentState
from agent.retrieval_preview import RetrievalPreview
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from retrieval.models import Chunk, Document
from scripts.demo_evidence_export import offline_settings

QUERY = "retrieval evidence"
PANEL_TITLES = (
    "1. Inspect the unchanged baseline",
    "2. Collapse similar passages, not papers",
    "3. Apply quotas and minimums afterwards",
    "4. Review the measured limits",
)
OUTPUT_NAMES = (
    "baseline-preview.json",
    "collapsed-preview.json",
    "term-set-preview.json",
    "capped-preview.json",
    "failed-run.json",
    "failed-events.json",
    "checks.json",
    "transcript.txt",
)


def seed_corpus(container: AppContainer) -> None:
    """Index five synthetic passages, including a cross-paper copy and near-variant."""
    passages = (
        ("a-original", "paper-a", "retrieval evidence alpha beta gamma delta"),
        ("b-copy", "paper-b", "retrieval evidence alpha beta gamma delta"),
        ("c-variant", "paper-c", "retrieval evidence alpha beta gamma delta epsilon"),
        ("a-distinct", "paper-a", "retrieval evidence soil moisture remote sensing"),
        ("d-distinct", "paper-d", "retrieval evidence telescope stellar calibration"),
    )
    chunks = [
        Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title="Synthetic retrieval fixture",
            text=text,
            source="synthetic:collapse",
            metadata={"license": "synthetic-only"},
        )
        for chunk_id, document_id, text in passages
    ]
    documents = [
        Document(
            document_id=document_id,
            title=chunks[0].title,
            text="\n".join(chunk.text for chunk in chunks if chunk.document_id == document_id),
            source=chunks[0].source,
            metadata=chunks[0].metadata,
        )
        for document_id in dict.fromkeys(chunk.document_id for chunk in chunks)
    ]
    container.document_store.add_documents(documents, chunks)
    container.hybrid_retriever.add_chunks(chunks)
    container.graph_builder.index_chunks(chunks)


def run_demo(output_dir: Path) -> str:
    """Persist actual measurements and fail explicitly if any offline contract is broken."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new directory.")
    with (
        TemporaryDirectory(prefix="scholar-collapse-") as temporary,
        patch.object(
            httpx.HTTPTransport,
            "handle_request",
            side_effect=AssertionError("No external HTTP in this offline demo."),
        ) as sync_http,
        patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            side_effect=AssertionError("No external HTTP in this offline demo."),
        ) as async_http,
    ):
        application = create_app(
            offline_settings(Path(temporary) / "demo.sqlite3").model_copy(
                update={"max_source_docs": 5}
            )
        )
        container: AppContainer = application.state.container
        seed_corpus(container)
        corpus_before = container.document_store.list_chunks()
        with (
            TestClient(application) as client,
            patch.object(
                container.llm,
                "generate",
                side_effect=AssertionError("No live or fake model calls in this demo."),
            ) as generate,
            patch.object(
                container.event_log, "append_event", wraps=container.event_log.append_event
            ) as writes,
        ):

            def preview(**options: object) -> RetrievalPreview:
                response = client.post("/retrieve", json={"query": QUERY, **options})
                response.raise_for_status()
                return RetrievalPreview.model_validate_json(response.content)

            baseline = preview()
            collapsed = preview(near_duplicate_threshold=0.8)
            term_set = preview(near_duplicate_threshold=1)
            capped = preview(near_duplicate_threshold=0.8, max_chunks_per_document=1)
            preview_writes = writes.call_count
            response = client.post(
                "/query",
                json={"query": QUERY, "near_duplicate_threshold": 0.8, "min_evidence_documents": 4},
            )
            response.raise_for_status()
            failed = QueryResponse.model_validate_json(response.content).result
            response = client.get(f"/runs/{failed.run_id}/events")
            response.raise_for_status()
            events = response.json()
            if (
                failed.state != AgentState.ERROR
                or failed.answer is not None
                or len(events) != 6
                or events[-2]["event_type"] != "evidence_snapshot"
                or events[-1]["payload"]["payload"].get("code") != "insufficient_evidence_documents"
            ):
                raise RuntimeError("Expected an explicit pre-generation minimum-document failure.")
            snapshot = EvidenceSnapshot.model_validate(events[-2]["payload"])
            assessment = events[-1]["payload"]["payload"]["evidence_assessment"]
            originals = {source.chunk.chunk_id: source for source in baseline.sources}
            survivor_ranks = [originals[source.chunk.chunk_id].rank for source in collapsed.sources]
            exact_survivors = survivor_ranks == sorted(survivor_ranks) and all(
                source.chunk == originals[source.chunk.chunk_id].chunk
                and source.score == originals[source.chunk.chunk_id].score
                and source.path
                == [
                    *originals[source.chunk.chunk_id].path,
                    originals[source.chunk.chunk_id].retriever,
                ]
                and source.retriever == "near_duplicate_collapse"
                for source in collapsed.sources
            )
            failed_matches_preview = (
                snapshot.sources == collapsed.sources
                and snapshot.request.context == collapsed.context
                and snapshot.context_sha256 == collapsed.context_sha256
            )
            model_calls = generate.await_count
            http_calls = sync_http.call_count + async_http.call_count
            generation_records = sum(event["event_type"] == "generation_record" for event in events)
            corpus_unchanged = container.document_store.list_chunks() == corpus_before
            if (
                (len(baseline.sources), len(collapsed.sources), len(term_set.sources)) != (5, 3, 4)
                or not exact_survivors
                or not failed_matches_preview
                or not corpus_unchanged
                or model_calls
                or http_calls
                or preview_writes
                or generation_records
            ):
                raise RuntimeError("Measured collapse, provenance, or offline contract failed.")

    surviving_ids = [source.chunk.chunk_id for source in collapsed.sources]
    removed_ids = [key for key in originals if key not in surviving_ids]
    baseline_bytes = len(baseline.context.encode("utf-8"))
    collapsed_bytes = len(collapsed.context.encode("utf-8"))
    checks = {
        "baseline_chunks": len(baseline.sources),
        "baseline_documents": len({source.chunk.document_id for source in baseline.sources}),
        "baseline_context_bytes": baseline_bytes,
        "collapsed_chunks": len(collapsed.sources),
        "collapsed_documents": len({source.chunk.document_id for source in collapsed.sources}),
        "collapsed_context_bytes": collapsed_bytes,
        "term_set_threshold_chunks": len(term_set.sources),
        "capped_chunks": len(capped.sources),
        "capped_documents": len({source.chunk.document_id for source in capped.sources}),
        "surviving_chunk_ids": surviving_ids,
        "removed_chunk_ids": removed_ids,
        "exact_survivors": exact_survivors,
        "corpus_unchanged": corpus_unchanged,
        "failed_state": failed.state.value,
        "failed_assessment": assessment,
        "failed_snapshot_matches_preview": failed_matches_preview,
        "generation_records": generation_records,
        "model_calls": model_calls,
        "external_http_attempts": http_calls,
        "preview_event_writes": preview_writes,
    }
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"Omitted threshold: {checks['baseline_chunks']} chunks from "
                        f"{checks['baseline_documents']} documents.",
                        f"Exact context bytes (UTF-8): {baseline_bytes}.",
                        f"Threshold 1: {len(term_set.sources)} chunks; "
                        "equal TERM SETS, not identical text.",
                        "Bounded hybrid/graph retrieval and lexical reranking, "
                        "not semantic search.",
                        "These synthetic fixtures are not papers or scientific findings.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"Threshold 0.8: {len(collapsed.sources)} chunks from "
                        f"{checks['collapsed_documents']} documents.",
                        f"Removed from baseline: {', '.join(removed_ids)}.",
                        f"Context bytes: {baseline_bytes} -> {collapsed_bytes}.",
                        "Exact surviving chunks, scores, order, and upstream paths: "
                        f"{exact_survivors}.",
                        "Survivor retriever: near_duplicate_collapse. No candidate refill.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Collapse 0.8 then cap 1: {len(capped.sources)} chunks from "
                        f"{checks['capped_documents']} documents.",
                        f"Query, collapse 0.8 + minimum 4: {failed.state.value}; "
                        f"observed documents={assessment['observed_documents']}.",
                        f"Failed snapshot equals collapsed preview: {failed_matches_preview}.",
                        f"Model calls={model_calls}; generation records={generation_records}.",
                        "Read failed-events.json. HTTP 200 alone does not mean query success.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Preview event writes={preview_writes}; "
                        f"external HTTP attempts={http_calls}.",
                        f"Corpus unchanged={corpus_unchanged}; no live OR fake generation.",
                        "Frozen policy is retained in the failed run plan and events.",
                        "Lexical overlap is not semantic equivalence or evidence independence.",
                        "Important similar passages can be dropped. Inspect the originals.",
                        "Actual route measurements, not a latency or retrieval-quality benchmark.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    artifacts = {
        "baseline-preview.json": baseline.model_dump_json(indent=2).encode(),
        "collapsed-preview.json": collapsed.model_dump_json(indent=2).encode(),
        "term-set-preview.json": term_set.model_dump_json(indent=2).encode(),
        "capped-preview.json": capped.model_dump_json(indent=2).encode(),
        "failed-run.json": failed.model_dump_json(indent=2).encode(),
        "failed-events.json": response.content,
        "checks.json": (json.dumps(checks, sort_keys=True, indent=2) + "\n").encode(),
        "transcript.txt": transcript.encode(),
    }
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
