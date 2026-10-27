"""Measure screened-only retrieval and saved evidence using isolated synthetic data."""

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
from api.schemas import IngestResponse, QueryResponse
from llm.fake import FakeLLMAdapter
from llm.providers import HTTPProviderAdapter
from scripts.create_evidence_gif import render_panels
from scripts.demo_evidence_export import offline_settings
from storage.paper_collections import PaperCollection

ARTIFACT_NAMES = (
    "screened-preview.json",
    "screened-evidence.json",
    "screened-demo.json",
    "transcript.txt",
)


def _forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("This synthetic demo forbids external HTTP and live generation.")


def run_demo(output_dir: Path) -> str:
    """Save actual API output; never read ambient credentials or keep the temporary corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        target = output_dir / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}.")
    with (
        TemporaryDirectory(prefix="scholar-screened-selection-") as temporary,
        patch.object(httpx.HTTPTransport, "handle_request", _forbidden),
        patch.object(httpx.AsyncHTTPTransport, "handle_async_request", _forbidden),
        patch.object(HTTPProviderAdapter, "generate", _forbidden),
    ):
        database_path = Path(temporary) / "corpus.sqlite3"
        settings = offline_settings(database_path)
        application = create_app(settings)
        with TestClient(application) as client:
            ids = []
            for label in ("Included", "Excluded", "Unsure"):
                response = client.post(
                    "/ingest/text",
                    json={
                        "title": f"Synthetic {label} note",
                        "text": f"Synthetic GraphRAG {label} evidence, not a scientific finding.",
                        "source": "synthetic:screened-selection",
                    },
                )
                response.raise_for_status()
                ids.append(IngestResponse.model_validate_json(response.content).document_id)
            response = client.post(
                "/collections", json={"name": "Synthetic screened review", "document_ids": ids}
            )
            response.raise_for_status()
            collection = PaperCollection.model_validate_json(response.content)
            collection_path = f"/collections/{collection.collection_id}"
            for identifier, decision in zip(ids, ("include", "exclude", "unsure"), strict=True):
                client.put(
                    f"{collection_path}/screening/{identifier}",
                    json={
                        "collection_revision": 1,
                        "expected_decision_revision": 0,
                        "decision": decision,
                        "reason": f"Synthetic human {decision} opinion, not scientific validation.",
                    },
                ).raise_for_status()
            selected_before_restart = application.state.container.paper_screening.select_included(
                collection.collection_id, collection_revision=1
            )

        restarted = create_app(settings)
        container = restarted.state.container
        selected = container.paper_screening.select_included(
            collection.collection_id, collection_revision=1
        )
        body = {
            "query": "GraphRAG",
            "screened_collection": {
                "collection_id": collection.collection_id,
                "collection_revision": 1,
            },
        }
        with TestClient(restarted) as client:
            before_preview = database_path.read_bytes()
            with (
                patch.object(FakeLLMAdapter, "generate", _forbidden),
                patch.object(container.event_log, "append_event", _forbidden),
                patch.object(container.event_log, "append_transition", _forbidden),
            ):
                ordinary_response = client.post(
                    "/retrieve",
                    json={"query": "GraphRAG", "collection_id": collection.collection_id},
                )
                ordinary_response.raise_for_status()
                preview_response = client.post("/retrieve", json=body)
                preview_response.raise_for_status()
            ordinary = RetrievalPreview.model_validate_json(ordinary_response.content)
            preview = RetrievalPreview.model_validate_json(preview_response.content)
            ordinary_ids = sorted({source.chunk.document_id for source in ordinary.sources})
            preview_ids = sorted({source.chunk.document_id for source in preview.sources})
            previews_read_only = database_path.read_bytes() == before_preview
            preview_event_count = len(container.event_log.list_events())

            response = client.post("/query", json=body)
            response.raise_for_status()
            run = QueryResponse.model_validate_json(response.content).result
            if run.state.value != "DONE":
                raise RuntimeError(f"Synthetic query did not complete: {run.error}")
            export_path = f"/runs/{run.run_id}/export"
            evidence_response = client.get(export_path)
            evidence_response.raise_for_status()
            bundle = EvidenceBundle.model_validate_json(evidence_response.content)
            client.put(
                f"{collection_path}/screening/{ids[0]}",
                json={
                    "collection_revision": 1,
                    "expected_decision_revision": 1,
                    "decision": "exclude",
                    "reason": "A changed synthetic human opinion, not a scientific judgment.",
                },
            ).raise_for_status()
            before_rejection = database_path.read_bytes()
            with (
                patch.object(FakeLLMAdapter, "generate", _forbidden),
                patch.object(container.event_log, "append_event", _forbidden),
                patch.object(container.event_log, "append_transition", _forbidden),
            ):
                empty_preview = client.post("/retrieve", json=body)
                empty_query = client.post("/query", json=body)
            rejections_read_only = database_path.read_bytes() == before_rejection
            client.put(
                collection_path,
                json={
                    "name": "Revised synthetic criteria",
                    "document_ids": ids,
                    "expected_revision": 1,
                },
            ).raise_for_status()
            stale = client.post("/retrieve", json=body)
            stale_revision = stale.json().get("detail", {}).get("code")
            preserved = client.get(export_path)
            preserved.raise_for_status()
            unchanged_evidence = preserved.content == evidence_response.content

        with TestClient(create_app(settings)) as client:
            after_restart = client.get(export_path)
            after_restart.raise_for_status()
            unchanged_evidence = unchanged_evidence and (
                after_restart.content == evidence_response.content
            )
        if (
            selected != (ids[0],)
            or selected_before_restart != selected
            or ordinary_ids != sorted(ids)
            or preview_ids != [ids[0]]
            or preview.plan.observation.document_ids != selected
            or bundle.plan.observation.document_ids != selected
            or {source.chunk.document_id for source in bundle.snapshot.sources} != {ids[0]}
            or bundle.generation.provider != "fake"
            or not previews_read_only
            or preview_event_count != 0
            or not rejections_read_only
            or empty_preview.status_code != 409
            or empty_query.status_code != 409
            or stale.status_code != 409
            or stale_revision != "collection_revision_conflict"
            or not unchanged_evidence
        ):
            raise RuntimeError("The measured screened-selection workflow failed its contract.")

    panels = [
        "\n".join(
            [
                "1. Choose a reviewed collection, not manual ID copying",
                f"Saved members: {len(ids)}; current human includes: {len(selected)}",
                "One include, one exclude, one unsure: synthetic human opinions.",
                f"Restart preserves Python selection: {selected_before_restart == selected}",
                "No automatic labeling, paper-quality claim, or new connector.",
            ]
        ),
        "\n".join(
            [
                "2. Preview the actual screened-only evidence",
                f"Ordinary collection_id: {len(ordinary_ids)} evidence documents",
                f"Opt-in screened_collection: {len(preview_ids)} evidence documents",
                f"Preview database unchanged: {previews_read_only}",
                f"Agent events after previews: {preview_event_count}",
                "Both fake/live generation and event appends were forbidden for previews.",
            ]
        ),
        "\n".join(
            [
                "3. Save a deliberately fake answer with exact source IDs",
                f"Run state: {bundle.status}; provider: {bundle.generation.provider}",
                f"Selected IDs retained in the saved plan: {len(selected)}",
                f"Frozen evidence documents: {len(bundle.snapshot.sources)}",
                "Only the query uses the offline fake; it does not summarize research.",
                "External HTTP and live generation were forbidden throughout.",
            ]
        ),
        "\n".join(
            [
                "4. Fail closed when the review changes",
                f"Empty selection: preview HTTP {empty_preview.status_code}; "
                f"query HTTP {empty_query.status_code}",
                f"Rejected requests leave database unchanged: {rejections_read_only}",
                f"Outdated collection revision: HTTP {stale.status_code}",
                f"Saved evidence unchanged after re-screening and restart: {unchanged_evidence}",
                "Later requests see current labels; a prior selection freezes IDs only.",
            ]
        ),
    ]
    transcript = "\n\n".join(panels) + "\n"
    measurements = {
        "synthetic_only": True,
        "request": body,
        "ordinary_document_ids": ordinary_ids,
        "screened_document_ids": preview_ids,
        "python_selected_document_ids": selected,
        "selection_survived_restart": selected_before_restart == selected,
        "previews_read_only": previews_read_only,
        "preview_event_count": preview_event_count,
        "generation_provider": bundle.generation.provider,
        "saved_state": bundle.status,
        "empty_preview_status": empty_preview.status_code,
        "empty_query_status": empty_query.status_code,
        "rejections_read_only": rejections_read_only,
        "stale_revision_status": stale.status_code,
        "saved_evidence_unchanged": unchanged_evidence,
    }
    (output_dir / "screened-preview.json").write_bytes(preview_response.content)
    (output_dir / "screened-evidence.json").write_bytes(evidence_response.content)
    (output_dir / "screened-demo.json").write_text(
        json.dumps(measurements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def create_gif(transcript: str, output_path: Path) -> None:
    """Illustrate four measured panels without replacing an existing artifact."""
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_path}.")
    render_panels(
        transcript.strip().split("\n\n"),
        output_path,
        banner="SCHOLAR RAG / SCREENED EVIDENCE / SYNTHETIC + OFFLINE",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif", type=Path)
    arguments = parser.parse_args()
    if arguments.gif is not None and (arguments.gif.exists() or arguments.gif.is_symlink()):
        raise FileExistsError(f"Refusing to overwrite {arguments.gif}.")
    transcript = run_demo(arguments.output)
    if arguments.gif is not None:
        create_gif(transcript, arguments.gif)
    print(transcript, end="")


if __name__ == "__main__":
    main()
