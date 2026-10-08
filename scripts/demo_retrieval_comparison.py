"""Measure side-by-side retrieval policies through the real API, wholly offline."""

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from agent.retrieval_comparison import RetrievalComparison, RetrievalComparisonRequest
from api.application import create_app
from api.dependencies import AppContainer
from llm.fake import FakeLLMAdapter
from llm.router import RoutingLLMAdapter
from scripts.demo_evidence_export import offline_settings
from scripts.demo_near_duplicate_evidence import seed_corpus
from storage.event_log import SQLiteEventLog

PANEL_TITLES = (
    "1. Compare before generating",
    "2. Inspect exact changes",
    "3. Retain the failed minimum assessment",
    "4. Save current-corpus inspection artifacts",
)
OUTPUT_NAMES = ("request.json", "comparison.json", "comparison.md", "checks.json", "transcript.txt")


def run_demo(output_dir: Path) -> str:
    """Save actual API output, measured checks and a transcript; refuse artifact collisions."""
    for name in OUTPUT_NAMES:
        target = output_dir / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}; choose a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    request = RetrievalComparisonRequest(
        query="retrieval evidence",
        document_ids=("paper-a", "paper-b", "paper-c", "paper-d"),
        baseline={},
        candidate={
            "evidence_policy": {
                "near_duplicate_threshold": 0.8,
                "max_chunks_per_document": 1,
                "min_evidence_documents": 4,
            }
        },
    )
    payload = request.model_dump(mode="json", exclude_none=True)
    with (
        TemporaryDirectory(prefix="scholar-retrieval-comparison-") as temporary,
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
        patch.object(
            RoutingLLMAdapter, "generate", side_effect=AssertionError("No model generation.")
        ) as routed,
        patch.object(
            FakeLLMAdapter, "generate", side_effect=AssertionError("No fake generation either.")
        ) as fake,
        patch.object(
            SQLiteEventLog, "append_event", side_effect=AssertionError("No agent events.")
        ) as events,
        patch.object(
            SQLiteEventLog, "append_transition", side_effect=AssertionError("No state transitions.")
        ) as transitions,
    ):
        database = Path(temporary) / "comparison.sqlite3"
        settings = offline_settings(database).model_copy(
            update={"max_source_docs": 5, "max_hops": 1}
        )
        application = create_app(settings)
        container: AppContainer = application.state.container
        seed_corpus(container)
        before = database.read_bytes()
        with (
            TestClient(application) as client,
            patch.object(container.runner, "preview", wraps=container.runner.preview) as preview,
        ):
            response = client.post("/research/compare-retrieval", json=payload)
            response.raise_for_status()
            preview_calls = preview.await_count
            result = RetrievalComparison.model_validate_json(response.content)
            markdown_response = client.post(
                "/research/compare-retrieval", params={"format": "markdown"}, json=payload
            )
            markdown_response.raise_for_status()
            python_result = asyncio.run(container.retrieval_comparator.compare(request))
            agent_events = len(container.event_log.list_events())
        with TestClient(create_app(settings)) as client:
            restarted = client.post("/research/compare-retrieval", json=payload)
            restarted.raise_for_status()
        counts = result.evidence.statistics
        assessment = result.candidate.preview.evidence_assessment
        checks = {
            "comparison_preview_calls": preview_calls,
            "baseline_chunks": result.baseline.source_count,
            "candidate_chunks": result.candidate.source_count,
            "baseline_documents": result.baseline.document_count,
            "candidate_documents": result.candidate.document_count,
            "shared_chunks": counts.shared_count,
            "added_chunks": counts.added_count,
            "removed_chunks": counts.removed_count,
            "baseline_context_bytes": result.baseline.context_utf8_bytes,
            "candidate_context_bytes": result.candidate.context_utf8_bytes,
            "context_bytes_delta": result.context_bytes_delta,
            "baseline_context_sha256": result.baseline.preview.context_sha256,
            "candidate_context_sha256": result.candidate.preview.context_sha256,
            "model_calls": routed.call_count + fake.call_count,
            "external_http_attempts": sync_http.call_count + async_http.call_count,
            "event_writes": events.call_count + transitions.call_count,
            "agent_events": agent_events,
            "database_unchanged": before == database.read_bytes(),
            "python_http_equal": python_result == result,
            "restart_equal": restarted.content == response.content,
            "json_bytes": len(response.content),
            "markdown_bytes": len(markdown_response.content),
        }
        if (
            (result.baseline.source_count, result.candidate.source_count) != (5, 2)
            or (result.baseline.document_count, result.candidate.document_count) != (4, 2)
            or (counts.shared_count, counts.added_count, counts.removed_count) != (2, 0, 3)
            or assessment is None
            or assessment.passed
            or assessment.observed_documents != 2
            or result.source_changes.text_changed != 0
            or result.source_changes.retriever_changed != 2
            or preview_calls != 2
            or any(
                checks[key]
                for key in ("model_calls", "external_http_attempts", "event_writes", "agent_events")
            )
            or not all(
                checks[key] for key in ("database_unchanged", "python_http_equal", "restart_equal")
            )
            or result.to_json() != response.text
            or result.to_markdown() != markdown_response.text
        ):
            raise RuntimeError("The measured retrieval comparison or offline contract failed.")
    checks["temporary_database_removed"] = not database.exists()
    if not checks["temporary_database_removed"]:
        raise RuntimeError("The temporary demonstration database was not removed.")
    baseline, candidate = result.baseline, result.candidate
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"POST /research/compare-retrieval -> HTTP {response.status_code}",
                        "One shared query/scope; baseline defaults vs "
                        "collapse 0.8 + cap 1 + minimum 4.",
                        f"Actual preview calls for one comparison: {preview_calls}",
                        f"Chunks: {baseline.source_count} -> {candidate.source_count}; "
                        f"documents: {baseline.document_count} -> {candidate.document_count}",
                        "Five synthetic passages, not publications or scientific findings.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"Shared: {counts.shared_count}; added: {counts.added_count}; "
                        f"removed: {counts.removed_count}",
                        f"Context bytes (UTF-8): {baseline.context_utf8_bytes} -> "
                        f"{candidate.context_utf8_bytes}; delta: {result.context_bytes_delta}",
                        f"Shared text changed: {result.source_changes.text_changed}; "
                        f"retriever changed: {result.source_changes.retriever_changed}",
                        f"Context digest changed: {result.changes.context_changed}",
                        "Full previews retain exact passages, ranks, scores, paths, "
                        "policies and limits.",
                        "Identity overlap and smaller context are not "
                        "retrieval-quality measurements.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Required documents: {assessment.required_documents}; "
                        f"observed: {assessment.observed_documents}; passed: {assessment.passed}",
                        "The candidate still returns its complete captured evidence for review.",
                        f"Model calls: {checks['model_calls']}; external HTTP attempts: "
                        f"{checks['external_http_attempts']}; "
                        f"event writes: {checks['event_writes']}",
                        f"Database unchanged: {checks['database_unchanged']}",
                        "No /query, fake answer, saved run, or scientific-support judgment.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Python / HTTP equal: {checks['python_http_equal']}; "
                        f"unchanged-corpus restart equal: {checks['restart_equal']}",
                        f"JSON: {checks['json_bytes']} bytes; "
                        f"Markdown: {checks['markdown_bytes']} bytes",
                        f"Temporary database removed: {checks['temporary_database_removed']}",
                        "Saved request, JSON, Markdown, checks and measured transcript.",
                        "Two awaits can see corpus changes: not a frozen or controlled experiment.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    artifacts = {
        "request.json": json.dumps(payload, indent=2, sort_keys=True) + "\n",
        "comparison.json": response.text,
        "comparison.md": markdown_response.text,
        "checks.json": json.dumps(checks, indent=2, sort_keys=True) + "\n",
        "transcript.txt": transcript,
    }
    for name, content in artifacts.items():
        with (output_dir / name).open("x", encoding="utf-8") as destination:
            destination.write(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
