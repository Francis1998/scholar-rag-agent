"""Execute collection browsing against original synthetic notes with forbidden-work guards."""

import argparse
import hashlib
import json
import socket
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from urllib.parse import urlencode, urlsplit

from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Chunk, Document
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from storage.paper_collections import PaperCollection
from storage.paper_screening import ScreeningExport, ScreeningSubmission

SELECTED_ID = "paper/../synthetic-\u7814?draft#v1"
PANEL_TITLES = (
    "1. Discover a saved collection",
    "2. Inspect human status, not automatic inclusion",
    "3. Follow exact readers and revision-bound downloads",
    "4. Restart and measure the read-only boundary",
)


def _seed(container: AppContainer) -> PaperCollection:
    identifiers = (SELECTED_ID, "background", "uncertain", "earlier", "unread")
    documents = [
        Document(
            document_id=identifier,
            title=f"Synthetic methods note {index}",
            source="synthetic:collection-browser",
            text="Original synthetic fixture, not a publication or scientific finding.",
        )
        for index, identifier in enumerate((*identifiers, "outside"))
    ]
    container.document_store.add_documents(
        documents,
        [
            Chunk(
                **document.model_dump(exclude={"text", "metadata"}),
                chunk_id=f"{document.document_id}/chunk",
                text="Synthetic literal phrase. <script>remains text</script>.",
                metadata={"chunk_index": "0"},
            )
            for document in documents
        ],
    )
    collection = container.paper_collections.create(
        name="Synthetic human review", document_ids=identifiers
    )
    container.paper_screening.submit(
        collection.collection_id,
        "earlier",
        ScreeningSubmission(
            collection_revision=1,
            expected_decision_revision=0,
            decision="include",
            reason="Earlier synthetic human opinion; not a current include.",
        ),
    )
    collection = container.paper_collections.replace(
        collection.collection_id,
        name=collection.name,
        document_ids=identifiers,
        expected_revision=1,
    )
    for identifier, decision in zip(identifiers, ("include", "exclude", "unsure"), strict=False):
        container.paper_screening.submit(
            collection.collection_id,
            identifier,
            ScreeningSubmission.model_validate(
                {
                    "collection_revision": 2,
                    "expected_decision_revision": 0,
                    "decision": decision,
                    "reason": f"Synthetic human {decision} reason, not scientific validation.",
                }
            ),
        )
    return collection


def _extra_guards(stack: ExitStack, container: AppContainer) -> list[Mock]:
    return [
        stack.enter_context(
            patch.object(target, method, side_effect=AssertionError("Forbidden browser work."))
        )
        for target, method in (
            (container.event_log, "append_transition"),
            (container.paper_screening, "submit"),
            (container.worksheets, "build"),
            (socket, "create_connection"),
            (socket.socket, "connect"),
        )
    ]


def run_demo(output_dir: Path) -> str:
    """Save actual response bytes and checks, not fabricated browser screenshots."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    artifacts: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    with TemporaryDirectory(prefix="collection-browser-") as temporary:
        database = Path(temporary) / "corpus.sqlite3"
        settings = offline_settings(database)
        app = create_app(settings)
        container: AppContainer = app.state.container
        collection = _seed(container)
        before = database.read_bytes()
        with (
            TestClient(app) as client,
            browsing_guards(container) as guards,
            ExitStack() as stack,
        ):
            extra = _extra_guards(stack, container)

            def get(name: str, url: str) -> SearchPage:
                parts = urlsplit(url)
                if (
                    parts.scheme
                    or parts.netloc
                    or not parts.path.startswith(("/explore", "/collections/"))
                ):
                    raise RuntimeError("Navigation left the local read-only workflow.")
                response = client.get(url)
                response.raise_for_status()
                if response.headers["cache-control"] != "no-store":
                    raise RuntimeError("A browser response lost its privacy header.")
                artifacts[name] = response.content
                return SearchPage(response.text)

            catalog = get("catalog.html", "/explore?limit=1&source=synthetic%3Acollection-browser")
            listed = get("collections.html", catalog.links["saved-collections"])
            detail_url = listed.links["inspect-collection-1"]
            detail = get("collection.html", detail_url)
            filtered = get(
                "stale.html",
                detail.form_action
                + "?"
                + urlencode({**detail.form_values, "status": "stale", "limit": "1"}),
            )
            if filtered.documents != ["earlier"]:
                raise RuntimeError("Status filtering changed the membership or lost the stale row.")
            paper = get("paper.html", detail.links["inspect-member-1"])
            source = get("source.html", paper.links["source-context-1"])
            search = get("search-form.html", detail.links["search-collection"])
            worksheet = get("worksheet-form.html", detail.links["worksheet-collection"])
            if (
                paper.documents != [SELECTED_ID]
                or source.chunks != [f"{SELECTED_ID}/chunk"]
                or any(
                    page.links["back-to-collection"] != detail_url
                    for page in (paper, source, search, worksheet)
                )
            ):
                raise RuntimeError("Exact identities or native return navigation changed.")
            for link, filename in (
                ("screening-json", "screening.json"),
                ("screening-csv", "screening.csv"),
                ("duplicates-json", "duplicates.json"),
                ("duplicates-markdown", "duplicates.md"),
            ):
                get(filename, detail.links[f"download-{link}"])
            snapshot = ScreeningExport.model_validate_json(artifacts["screening.json"])
            for format in ("json", "csv"):
                expected = container.paper_screening.export_results(
                    collection.collection_id, collection_revision=2, format=format
                )
                if expected.content != artifacts[f"screening.{format}"]:
                    raise RuntimeError("Python and HTTP screening exports differ.")
            mismatch = client.get(
                "/explore/collection",
                params={"collection_id": collection.collection_id, "collection_revision": 1},
            )
            counts.update({name: guard.call_count for name, guard in guards.items()})
            counts.update({f"extra-{index}": guard.call_count for index, guard in enumerate(extra)})
        restarted = create_app(settings)
        with (
            TestClient(restarted) as client,
            browsing_guards(restarted.state.container) as guards,
            ExitStack() as stack,
        ):
            extra = _extra_guards(stack, restarted.state.container)
            reopened = client.get(detail_url)
            reopened.raise_for_status()
            artifacts["restarted.html"] = reopened.content
            counts.update({f"restart:{name}": guard.call_count for name, guard in guards.items()})
            counts.update(
                {f"restart:extra-{index}": guard.call_count for index, guard in enumerate(extra)}
            )
        checks = {
            "collection_revision": snapshot.collection_revision,
            "members": snapshot.total_documents,
            "counts": snapshot.counts.model_dump(),
            "current_include_ids": list(snapshot.included_document_ids),
            "exact_source_and_return": True,
            "python_http_exports_equal": True,
            "mismatched_revision_status": mismatch.status_code,
            "restart_html_identical": artifacts["collection.html"] == artifacts["restarted.html"],
            "restart_sha256": {
                name: hashlib.sha256(artifacts[name]).hexdigest()
                for name in ("collection.html", "restarted.html")
            },
            "database_bytes_unchanged": database.read_bytes() == before,
            "event_count": len(container.event_log.list_events()),
            "forbidden_call_counts": counts,
            "output_bytes": {name: len(content) for name, content in artifacts.items()},
        }
        if (
            any(counts.values())
            or checks["event_count"] != 0
            or mismatch.status_code != 409
            or snapshot.total_documents != 5
            or set(snapshot.counts.model_dump().values()) != {1}
            or snapshot.included_document_ids != (SELECTED_ID,)
            or not checks["restart_html_identical"]
            or not checks["database_bytes_unchanged"]
        ):
            raise RuntimeError("Measured collection-browser contract failed.")
    checks["temporary_database_removed"] = not database.exists()
    if not checks["temporary_database_removed"]:
        raise RuntimeError("Temporary corpus was not removed.")
    transcript = (
        "\n\n".join(
            (
                f"{PANEL_TITLES[0]}\n"
                "Executed /explore -> saved collections -> exact collection revision.\n"
                f"Current revision: {snapshot.collection_revision}; "
                f"members: {snapshot.total_documents}.\n"
                "Membership counts are not screening decisions.",
                f"{PANEL_TITLES[1]}\n"
                + "; ".join(
                    f"{key}: {value}" for key, value in snapshot.counts.model_dump().items()
                )
                + ".\nFiltered stale row retained its earlier include reason.\n"
                "Only 1 current human include; no automatic screening or selection.",
                f"{PANEL_TITLES[2]}\n"
                "Executed paper -> exact source -> return; scope forms retained the collection.\n"
                "Downloaded actual screening JSON / CSV and duplicate JSON / Markdown.\n"
                f"Python / HTTP bytes equal; mismatched revision: HTTP {mismatch.status_code}.\n"
                "Separate reads are not frozen together; duplicate groups are candidates.",
                f"{PANEL_TITLES[3]}\n"
                f"Restart HTML identical: {checks['restart_html_identical']}.\n"
                f"Database unchanged: {checks['database_bytes_unchanged']}; "
                f"events: {checks['event_count']}; forbidden calls: {sum(counts.values())}.\n"
                "Temporary fixture removed. No live or fake generation; no external network.\n"
                "Executed synthetic output illustration, not a recording or benchmark.",
            )
        )
        + "\n"
    )
    artifacts["checks.json"] = (json.dumps(checks, indent=2, ensure_ascii=True) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode()
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_demo(args.output_dir), end="")


if __name__ == "__main__":
    main()
