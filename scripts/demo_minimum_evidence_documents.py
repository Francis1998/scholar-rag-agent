"""Measure minimum-document diagnostics and generation admission using synthetic offline data."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from agent.comparison_models import RunComparison
from agent.evidence import EvidenceBundle, EvidenceSnapshot
from agent.models import AgentRunResult, AgentState
from agent.retrieval_preview import RetrievalPreview
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import QueryResponse
from retrieval.evidence_policy import EvidenceAssessment
from scripts.demo_evidence_export import offline_settings
from scripts.demo_per_paper_evidence_limits import QUERY, seed_corpus

PANEL_TITLES = (
    "1. Count documents, not passages",
    "2. Stop before answer generation",
    "3. Inspect a sufficient selection",
    "4. Preserve the requested threshold",
)
OUTPUT_NAMES = (
    "single-preview.json",
    "unknown-preview.json",
    "sufficient-preview.json",
    "failed-run.json",
    "failed-events.json",
    "bundle.json",
    "bundle.md",
    "comparison.json",
    "checks.json",
    "transcript.txt",
)


def _query(client: TestClient, **options: object) -> AgentRunResult:
    response = client.post("/query", json={"query": QUERY, **options})
    response.raise_for_status()
    return QueryResponse.model_validate_json(response.content).result


def run_demo(output_dir: Path) -> str:
    """Exercise real routes and persist measured output; never open an ambient corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new directory.")
    with (
        TemporaryDirectory(prefix="scholar-minimum-evidence-") as temporary,
        patch.object(
            httpx.HTTPTransport,
            "handle_request",
            side_effect=AssertionError("The synthetic demo must not use external HTTP."),
        ) as sync_http,
        patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            side_effect=AssertionError("The synthetic demo must not use external HTTP."),
        ) as async_http,
    ):
        settings = offline_settings(Path(temporary) / "demo.sqlite3").model_copy(
            update={"max_source_docs": 6}
        )
        application = create_app(settings)
        container: AppContainer = application.state.container
        seed_corpus(container)
        with (
            TestClient(application) as client,
            patch.object(container.llm, "generate", wraps=container.llm.generate) as generate,
            patch.object(
                container.event_log, "append_event", wraps=container.event_log.append_event
            ) as writes,
        ):
            preview_generations, preview_writes = 0, 0

            def preview(**options: object) -> RetrievalPreview:
                nonlocal preview_generations, preview_writes
                before_generations, before_writes = generate.await_count, writes.call_count
                response = client.post("/retrieve", json={"query": QUERY, **options})
                response.raise_for_status()
                preview_generations += generate.await_count - before_generations
                preview_writes += writes.call_count - before_writes
                return RetrievalPreview.model_validate_json(response.content)

            single = preview(document_ids=["paper-a"], min_evidence_documents=2)
            unknown = preview(document_ids=["not-ingested"], min_evidence_documents=1)
            before_failure = generate.await_count
            failed = _query(client, document_ids=["paper-a"], min_evidence_documents=2)
            failed_generations = generate.await_count - before_failure
            failed_events_path = f"/runs/{failed.run_id}/events"
            failed_events_response = client.get(failed_events_path)
            failed_events_response.raise_for_status()
            failed_events = failed_events_response.json()
            snapshots = [
                event for event in failed_events if event["event_type"] == "evidence_snapshot"
            ]
            if failed.state != AgentState.ERROR or failed.answer is not None or len(snapshots) != 1:
                raise RuntimeError("The insufficient query must save evidence and fail explicitly.")
            failed_snapshot = EvidenceSnapshot.model_validate(snapshots[0]["payload"])
            diagnostic = failed_events[-1]["payload"]["payload"]
            failed_assessment = EvidenceAssessment.model_validate(diagnostic["evidence_assessment"])
            generation_records = sum(
                event["event_type"] == "generation_record" for event in failed_events
            )
            done_events = sum(
                event["event_type"] == "state_transition" and event["payload"]["to_state"] == "DONE"
                for event in failed_events
            )
            if (
                diagnostic["code"] != "insufficient_evidence_documents"
                or failed_assessment != single.evidence_assessment
                or failed_snapshot.sources != single.sources
                or failed_snapshot.request.context != single.context
                or failed_generations
                or generation_records
                or done_events
            ):
                raise RuntimeError(
                    "Failure must retain the exact previewed evidence without generation."
                )

            collection = client.post(
                "/collections",
                json={"name": "Synthetic selection", "document_ids": ["paper-a", "paper-b"]},
            )
            collection.raise_for_status()
            selection = {
                "collection_id": collection.json()["collection_id"],
                "max_chunks_per_document": 1,
            }
            sufficient = preview(**selection, min_evidence_documents=2)
            accepted = _query(client, **selection, min_evidence_documents=2)
            lower = _query(client, **selection, min_evidence_documents=1)
            if accepted.state != AgentState.DONE or lower.state != AgentState.DONE:
                raise RuntimeError("Both sufficient requests must generate normally.")
            export_path = f"/runs/{accepted.run_id}/export"
            comparison_path = f"/runs/{lower.run_id}/compare/{accepted.run_id}"
            paths = (
                export_path,
                export_path + "?format=markdown",
                comparison_path,
                failed_events_path,
            )
            saved: dict[str, bytes] = {}
            for path in paths:
                response = client.get(path)
                response.raise_for_status()
                saved[path] = response.content
            bundle = EvidenceBundle.model_validate_json(saved[export_path])
            comparison = RunComparison.model_validate_json(saved[comparison_path])
            identical_context = (
                sufficient.sources == bundle.snapshot.sources
                and sufficient.context == bundle.snapshot.request.context
                and sufficient.context_sha256 == bundle.snapshot.context_sha256
            )
            policy_only_change = [
                key for key, changed in comparison.changes.model_dump().items() if changed
            ] == ["evidence_policy_changed"]
            if (
                single.evidence_assessment is None
                or unknown.evidence_assessment is None
                or sufficient.evidence_assessment is None
                or single.evidence_assessment.observed_documents != 1
                or single.evidence_assessment.passed
                or len(single.sources) != 3
                or unknown.evidence_assessment.observed_documents != 0
                or unknown.sources
                or not sufficient.evidence_assessment.passed
                or sufficient.evidence_assessment.observed_documents != 2
                or len(sufficient.sources) != 2
                or not identical_context
                or not policy_only_change
                or not comparison.any_changes
                or bundle.generation.provider != "fake"
                or preview_generations
                or preview_writes
                or generate.await_count != 2
            ):
                raise RuntimeError(
                    "Measured count, no-generation, or saved-provenance contract failed."
                )
            query_generations = generate.await_count
            before_events = container.event_log.list_events()

        reopened = create_app(settings)
        restarted: AppContainer = reopened.state.container
        with (
            TestClient(reopened) as client,
            patch.object(
                restarted.llm,
                "generate",
                side_effect=AssertionError("No generation on restart reads."),
            ),
            patch.object(
                restarted.runner._executor,
                "retrieve",
                side_effect=AssertionError("No retrieval on restart reads."),
            ),
            patch.object(
                restarted.event_log,
                "append_event",
                side_effect=AssertionError("No event writes on restart reads."),
            ),
        ):
            matches = []
            for path, content in saved.items():
                response = client.get(path)
                response.raise_for_status()
                matches.append(response.content == content)
            restart_identical = all(matches)
            if not restart_identical or restarted.event_log.list_events() != before_events:
                raise RuntimeError(
                    "Saved failed evidence, exports, or comparison changed on restart."
                )
        live_attempts = sync_http.call_count + async_http.call_count
        if live_attempts:
            raise RuntimeError("The offline demonstration attempted an external HTTP call.")

    checks = {
        "single_chunks": len(single.sources),
        "single_assessment": single.evidence_assessment.model_dump(mode="json"),
        "unknown_assessment": unknown.evidence_assessment.model_dump(mode="json"),
        "failed_state": failed.state.value,
        "failed_generations": failed_generations,
        "failed_snapshot_chunks": len(failed_snapshot.sources),
        "failed_generation_records": generation_records,
        "failed_done_events": done_events,
        "sufficient_assessment": sufficient.evidence_assessment.model_dump(mode="json"),
        "sufficient_chunks": len(sufficient.sources),
        "preview_generations": preview_generations,
        "preview_event_writes": preview_writes,
        "preview_matches_query": identical_context,
        "policy_only_change": policy_only_change,
        "restart_byte_identical": restart_identical,
        "query_fake_generations": query_generations,
        "live_http_attempts": live_attempts,
    }
    artifacts = {
        "single-preview.json": single.model_dump_json(indent=2).encode(),
        "unknown-preview.json": unknown.model_dump_json(indent=2).encode(),
        "sufficient-preview.json": sufficient.model_dump_json(indent=2).encode(),
        "failed-run.json": failed.model_dump_json(indent=2).encode(),
        "failed-events.json": saved[failed_events_path],
        "bundle.json": saved[export_path],
        "bundle.md": saved[export_path + "?format=markdown"],
        "comparison.json": saved[comparison_path],
        "checks.json": (json.dumps(checks, sort_keys=True, indent=2) + "\n").encode(),
    }
    for name, content in artifacts.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(content)
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"Single-paper preview: chunks={len(single.sources)}; "
                        f"distinct documents={single.evidence_assessment.observed_documents}.",
                        f"Required={single.evidence_assessment.required_documents}; "
                        f"passed={single.evidence_assessment.passed}. Evidence stays inspectable.",
                        "Unknown selected ID: "
                        f"observed={unknown.evidence_assessment.observed_documents}; "
                        f"passed={unknown.evidence_assessment.passed}. No widened search.",
                        "Counts use final captured document IDs, not titles or selected IDs.",
                        "Synthetic fixtures only; document counts are not scientific support.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        f"Query state={failed.state.value}; "
                        f"answer present={failed.answer is not None}.",
                        f"Required={failed_assessment.required_documents}; "
                        f"observed={failed_assessment.observed_documents}.",
                        f"Failed-query generation calls={failed_generations}.",
                        f"Saved snapshot chunks={len(failed_snapshot.sources)}; "
                        f"generation records={generation_records}; DONE events={done_events}.",
                        "Failure / preview context identical="
                        f"{failed_snapshot.request.context == single.context}.",
                        "Read the diagnostic and evidence in failed-events.json.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Collection paper-a + paper-b, cap 1: chunks={len(sufficient.sources)}.",
                        f"Required={sufficient.evidence_assessment.required_documents}; "
                        f"observed={sufficient.evidence_assessment.observed_documents}; "
                        f"passed={sufficient.evidence_assessment.passed}.",
                        f"Query state={bundle.status}; provider={bundle.generation.provider}.",
                        f"Preview / query context identical={identical_context}.",
                        f"Context SHA-256: {sufficient.context_sha256}",
                        "The fake answer demonstrates plumbing, not a scientific conclusion.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Minimum 1 vs 2: context equal={not comparison.changes.context_changed}; "
                        f"policy-only change={policy_only_change}.",
                        "Failed events, exports, comparison after restart: "
                        f"identical={restart_identical}.",
                        f"Preview generation calls={preview_generations}; "
                        f"event writes={preview_writes}.",
                        f"Query fake generations={query_generations}; "
                        f"live HTTP attempts={live_attempts}.",
                        "Read checks.json and the full saved artifacts before sharing.",
                        "No refill, retry, relaxed capture bounds, or rewritten source provenance.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    with (output_dir / "transcript.txt").open("x", encoding="utf-8") as stream:
        stream.write(transcript)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
