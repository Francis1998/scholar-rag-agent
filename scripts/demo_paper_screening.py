"""Measure human screening, restart, scoped preview and stale-label protection offline."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from api.application import create_app
from llm.fake import FakeLLMAdapter
from llm.providers import HTTPProviderAdapter
from scripts.create_evidence_gif import render_panels
from scripts.demo_evidence_export import offline_settings
from storage.paper_screening import ScreeningQueue


def _forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("This synthetic demo forbids external HTTP and answer generation.")


def _queue(client: TestClient, path: str, revision: int) -> ScreeningQueue:
    response = client.get(path, params={"collection_revision": revision})
    response.raise_for_status()
    return ScreeningQueue.model_validate_json(response.content)


def run_demo(output_dir: Path) -> str:
    """Persist measured artifacts; temporary synthetic SQLite is removed on exit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("screening.json", "transcript.txt"):
        target = output_dir / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}.")
    with (
        TemporaryDirectory(prefix="scholar-screening-") as temporary,
        patch.object(httpx.HTTPTransport, "handle_request", _forbidden),
        patch.object(httpx.AsyncHTTPTransport, "handle_async_request", _forbidden),
        patch.object(FakeLLMAdapter, "generate", _forbidden),
        patch.object(HTTPProviderAdapter, "generate", _forbidden),
    ):
        settings = offline_settings(Path(temporary) / "corpus.sqlite3")
        with TestClient(create_app(settings)) as client:
            ids = []
            for title in ("Methods", "Background", "Comparison"):
                response = client.post(
                    "/ingest/text",
                    json={
                        "title": f"Synthetic {title}",
                        "text": f"Synthetic note, not a publication. GraphRAG {title} evidence.",
                        "source": "synthetic:screening",
                    },
                )
                response.raise_for_status()
                ids.append(response.json()["document_id"])
            created = client.post(
                "/collections", json={"name": "Synthetic reading queue", "document_ids": ids}
            )
            created.raise_for_status()
            collection_path = f"/collections/{created.json()['collection_id']}"
            path = f"{collection_path}/screening"
            initial = _queue(client, path, 1)
            for identifier, decision in zip(ids, ("include", "exclude", "unsure"), strict=True):
                client.put(
                    f"{path}/{identifier}",
                    json={
                        "collection_revision": 1,
                        "expected_decision_revision": 0,
                        "decision": decision,
                        "reason": f"Synthetic human {decision} opinion, not scientific validation.",
                    },
                ).raise_for_status()
            reviewed = _queue(client, path, 1)

        restarted = create_app(settings)
        with TestClient(restarted) as client:
            recovered = _queue(client, path, 1)
            preview = client.post(
                "/retrieve",
                json={"query": "GraphRAG", "document_ids": list(recovered.included_document_ids)},
            )
            preview.raise_for_status()
            preview_ids = {source["chunk"]["document_id"] for source in preview.json()["sources"]}
            client.put(
                collection_path,
                json={"name": "Revised criteria", "document_ids": ids, "expected_revision": 1},
            ).raise_for_status()
            stale = _queue(client, path, 2)
            stale_write = client.put(
                f"{path}/{ids[0]}",
                json={
                    "collection_revision": 1,
                    "expected_decision_revision": 1,
                    "decision": "include",
                    "reason": "An outdated collection revision must fail.",
                },
            )
            client.put(
                f"{path}/{ids[0]}",
                json={
                    "collection_revision": 2,
                    "expected_decision_revision": 1,
                    "decision": "include",
                    "reason": "Human re-screening against the revised collection.",
                },
            ).raise_for_status()
            renewed = _queue(client, path, 2)
            events = restarted.state.container.event_log.list_events()
        if (
            initial.counts.unscreened != 3
            or reviewed != recovered
            or recovered.included_document_ids != (ids[0],)
            or preview_ids != {ids[0]}
            or stale.counts.stale != 3
            or stale.included_document_ids
            or stale_write.status_code != 409
            or renewed.included_document_ids != (ids[0],)
            or events
        ):
            raise RuntimeError("The measured screening workflow did not satisfy its contract.")

    panels = [
        "\n".join(
            [
                "1. Start with an explicit reading queue",
                f"Collection members: {initial.total_documents}",
                f"Unscreened: {initial.counts.unscreened}",
                f"Included selection: {len(initial.included_document_ids)}",
                "Synthetic notes only; no automatic scientific judgment.",
            ]
        ),
        "\n".join(
            [
                "2. Save human labels and resume",
                f"Include: {recovered.counts.include}; exclude: {recovered.counts.exclude}",
                f"Unsure: {recovered.counts.unsure}",
                f"Restart preserves the complete queue: {reviewed == recovered}",
                "Reasons and decision revisions survive the restart.",
            ]
        ),
        "\n".join(
            [
                "3. Explicitly preview included papers",
                f"Selected document IDs: {len(recovered.included_document_ids)}",
                f"Documents in the actual preview: {len(preview_ids)}",
                f"Agent events written: {len(events)}",
                "External HTTP and both fake/live generation were forbidden.",
            ]
        ),
        "\n".join(
            [
                "4. Re-screen a changed collection",
                f"Stale decisions after revision change: {stale.counts.stale}",
                f"Old included selection: {len(stale.included_document_ids)}",
                f"Stale write HTTP status: {stale_write.status_code}",
                f"Explicitly re-screened selection: {len(renewed.included_document_ids)}",
            ]
        ),
    ]
    transcript = "\n\n".join(panels) + "\n"
    artifacts = {
        "synthetic_only": True,
        "initial": initial.model_dump(mode="json"),
        "reviewed": recovered.model_dump(mode="json"),
        "stale": stale.model_dump(mode="json"),
        "renewed": renewed.model_dump(mode="json"),
        "preview_document_ids": sorted(preview_ids),
        "stale_write_status": stale_write.status_code,
        "agent_event_count": len(events),
    }
    (output_dir / "screening.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    return transcript


def create_gif(transcript: str, output_path: Path) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_path}.")
    render_panels(
        transcript.strip().split("\n\n"),
        output_path,
        banner="SCHOLAR RAG / HUMAN SCREENING / SYNTHETIC + OFFLINE",
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
