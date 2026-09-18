"""Measure document-scoped API queries and graph traversal on synthetic offline data."""

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from agent.evidence import EvidenceBundle
from agent.models import AgentRunResult, AgentState
from api.application import create_app
from api.dependencies import AppContainer
from api.schemas import IngestResponse
from retrieval.hyde import HyDEExpander
from retrieval.multihop import MultiHopRetriever
from scripts.demo_evidence_export import offline_settings

QUERY = "retrieval"
PANEL_TITLES = (
    "1. Select papers in a synthetic, offline corpus",
    "2. Keep selection through every retrieval path",
    "3. Empty matches never widen the search",
    "4. Reopen SQLite and inspect saved evidence",
)


def _ingest(client: TestClient, title: str, text: str) -> IngestResponse:
    response = client.post(
        "/ingest/text", json={"title": title, "text": text, "source": "synthetic:scope-demo"}
    )
    response.raise_for_status()
    return IngestResponse.model_validate_json(response.content)


def _query(client: TestClient, document_ids: list[str] | None = None) -> AgentRunResult:
    payload: dict[str, object] = {"query": QUERY}
    if document_ids is not None:
        payload["document_ids"] = document_ids
    response = client.post("/query", json=payload)
    response.raise_for_status()
    result = AgentRunResult.model_validate(response.json()["result"])
    if result.state != AgentState.DONE:
        raise RuntimeError(f"Synthetic query failed: {result.error}")
    return result


def _export(client: TestClient, run_id: str) -> tuple[bytes, EvidenceBundle]:
    response = client.get(f"/runs/{run_id}/export?format=json")
    response.raise_for_status()
    bundle = EvidenceBundle.model_validate_json(response.content)
    if bundle.generation.provider != "fake":
        raise RuntimeError("The synthetic demo must use the offline fake provider.")
    return response.content, bundle


def _source_ids(bundle: EvidenceBundle) -> set[str]:
    return {source.chunk.document_id for source in bundle.snapshot.sources}


def run_demo(output_dir: Path) -> str:
    """Exercise actual API results with a temporary SQLite corpus and saved exports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("scoped.json", "scoped.md", "unscoped.json", "unknown.json", "transcript.txt")
    for name in names:
        if (output_dir / name).exists():
            raise FileExistsError(
                f"Refusing to overwrite {output_dir / name}; choose a new directory."
            )
    with TemporaryDirectory(prefix="scholar-document-scope-") as temporary:
        settings = offline_settings(Path(temporary) / "scope.sqlite3").model_copy(
            update={"max_source_docs": 2}
        )
        application = create_app(settings)
        container: AppContainer = application.state.container
        with TestClient(application) as client:
            expanded = asyncio.run(HyDEExpander().expand(QUERY))
            distractors = [
                _ingest(client, f"Synthetic high-ranking distractor {index}", expanded)
                for index in range(4)
            ]
            first = _ingest(
                client,
                "Synthetic selected methods",
                "Alpha selected retrieval. " + "unrelated " * 50,
            )
            second = _ingest(
                client,
                "Synthetic selected followup",
                "Beta Gamma followup retrieval. " + "unrelated " * 50,
            )
            bridge = _ingest(
                client, "Synthetic excluded bridge", "Alpha Beta excluded bridge with retrieval."
            )
            papers = [*distractors, first, second, bridge]
            selected = [first.document_id, second.document_id]
            supplied = [f" {first.document_id} ", second.document_id, first.document_id]

            unscoped_json, baseline = _export(client, _query(client).run_id)
            scoped_json, bundle = _export(client, _query(client, supplied).run_id)
            if _source_ids(baseline) & set(selected):
                raise RuntimeError("The demo's selected papers must be below the global top-2.")
            if (
                _source_ids(bundle) != set(selected)
                or {c.document_id for c in bundle.answer.citations} != set(selected)
                or bundle.plan.observation.document_ids != tuple(selected)
            ):
                raise RuntimeError("The API did not preserve the requested document scope.")
            path = f"/runs/{bundle.run_id}/export"
            markdown = client.get(path, params={"format": "markdown"})
            markdown.raise_for_status()

            graph = MultiHopRetriever(container.graph_store)
            graph_all = asyncio.run(graph.retrieve("Alpha", ["Alpha"], depth=3, limit=10))
            graph_scoped = asyncio.run(
                graph.retrieve("Alpha", ["Alpha"], depth=3, limit=10, document_ids=selected)
            )
            if {r.chunk.document_id for r in graph_all} != {
                first.document_id,
                bridge.document_id,
                second.document_id,
            } or {r.chunk.document_id for r in graph_scoped} != {first.document_id}:
                raise RuntimeError("The synthetic graph did not demonstrate the excluded bridge.")

            unknown_json, unknown = _export(client, _query(client, ["not-ingested"]).run_id)
            if (
                unknown.snapshot.sources
                or unknown.answer.citations
                or not unknown.answer.ungrounded
                or not unknown.answer.warnings
            ):
                raise RuntimeError(
                    "Unknown IDs must produce empty, explicitly ungrounded evidence."
                )
            empty = client.post("/query", json={"query": QUERY, "document_ids": []})
            null = client.post("/query", json={"query": QUERY, "document_ids": None})
            if empty.status_code != 422 or null.status_code != 422:
                raise RuntimeError("Invalid explicit scope must be rejected, never widened.")

            application.state.container = AppContainer(settings)
            _, reloaded = _export(client, _query(client, selected).run_id)
            if _source_ids(reloaded) != set(selected):
                raise RuntimeError("Reloaded indexes lost the requested document scope.")
            for fmt, original in (("json", scoped_json), ("markdown", markdown.content)):
                after = client.get(path, params={"format": fmt})
                after.raise_for_status()
                if after.content != original:
                    raise RuntimeError(f"{fmt} export changed after reopening SQLite.")
            initial = bundle.events[0].payload
            initial_payload = None if initial is None else initial.get("payload")
            if (
                not isinstance(initial_payload, dict)
                or initial_payload.get("document_ids") != selected
            ):
                raise RuntimeError("The initial event must record the effective scope.")

    (output_dir / "scoped.json").write_bytes(scoped_json)
    (output_dir / "scoped.md").write_bytes(markdown.content)
    (output_dir / "unscoped.json").write_bytes(unscoped_json)
    (output_dir / "unknown.json").write_bytes(unknown_json)
    transcript = (
        "\n\n".join(
            [
                "\n".join(
                    [
                        PANEL_TITLES[0],
                        f"Ingested papers: {len(papers)}; selected papers: {len(selected)}",
                        f"Provider: {bundle.generation.provider}; public listener: none",
                        "Global top-2 contains selected papers: "
                        f"{len(_source_ids(baseline) & set(selected))}",
                        f"Scoped top-2 contains selected papers: {len(_source_ids(bundle))}",
                        "Excluded documents in scoped context/citations: 0",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[1],
                        "HyDE + hash dense + BM25 -> RRF context: "
                        f"{len(bundle.snapshot.sources)} chunks",
                        f"Graph at Alpha: unscoped chunks={len(graph_all)}, "
                        f"scoped chunks={len(graph_scoped)}",
                        "Allowed tail via excluded bridge: blocked",
                        f"Normalized/deduplicated: {len(supplied)} supplied -> "
                        f"{len(selected)} effective",
                        "BM25 statistics remain global; selection is not authentication.",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[2],
                        f"Unknown ID -> state={unknown.status}; "
                        f"evidence={len(unknown.snapshot.sources)}; "
                        f"citations={len(unknown.answer.citations)}",
                        f"Answer ungrounded={unknown.answer.ungrounded}; "
                        f"warning recorded={bool(unknown.answer.warnings)}",
                        f"Empty list -> HTTP {empty.status_code}; null -> HTTP {null.status_code}",
                        "Unknown IDs retained in plan: "
                        f"{list(unknown.plan.observation.document_ids or ())}",
                    ]
                ),
                "\n".join(
                    [
                        PANEL_TITLES[3],
                        f"Reloaded scoped query: {len(_source_ids(reloaded))} selected papers",
                        "JSON and Markdown after restart: byte-identical",
                        f"Scope in plan and initial event: {len(selected)} IDs",
                        f"Context SHA-256: {bundle.snapshot.context_sha256}",
                        "Saved: scoped.json, scoped.md, unscoped.json, unknown.json, "
                        "transcript.txt",
                        "Temporary SQLite removed; saved evidence is not scientific proof.",
                    ]
                ),
            ]
        )
        + "\n"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def main() -> None:
    """Write synthetic exports and a measured transcript without any public listener."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("document-scope-demo"))
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
