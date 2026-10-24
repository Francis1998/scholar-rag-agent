"""Execute an offline synthetic human progress report through native navigation and downloads."""

import argparse
import hashlib
import json
import socket
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import parse_qsl, urlencode, urlsplit

from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from scripts.demo_corpus_explorer import _no_work, browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from storage.paper_collections import PaperCollection
from storage.paper_screening import ScreeningSubmission
from storage.screening_report import ScreeningReport, export_screening_report

MEMBER_IDS = (
    "included",
    "paper/../synthetic-\u7814?draft#v1",
    "same-reason",
    "case-different",
    "space-different",
    "unsure",
    "old-exclude",
    "old-include",
    "unread",
)
REASON = "Population mismatch\nHuman note: <literal>, not a finding. \u7814\u7a76"
PANEL_TITLES = (
    "1. Discover the saved human review",
    "2. Separate resolved from still needing review",
    "3. Audit exact exclusion reasons",
    "4. Save and reproduce without models or writes",
)
OUTPUT_NAMES = (
    "catalog.html",
    "collections.html",
    "collection.html",
    "report.json",
    "report.md",
    "snapshot.json",
    "stale-revision.json",
    "checks.json",
    "transcript.txt",
)


def _seed(container: AppContainer) -> PaperCollection:
    container.document_store.add_documents(
        [
            Document(
                document_id=identifier,
                title="Same synthetic title",
                source="synthetic:screening-report",
                text="Original synthetic fixture; not a publication or scientific finding.",
            )
            for identifier in MEMBER_IDS
        ],
        [],
    )
    collection = container.paper_collections.create(
        name="Synthetic screening progress", document_ids=MEMBER_IDS
    )
    for identifier, decision in ((MEMBER_IDS[6], "exclude"), (MEMBER_IDS[7], "include")):
        container.paper_screening.submit(
            collection.collection_id,
            identifier,
            ScreeningSubmission.model_validate(
                {
                    "collection_revision": 1,
                    "expected_decision_revision": 0,
                    "decision": decision,
                    "reason": REASON,
                }
            ),
        )
    collection = container.paper_collections.replace(
        collection.collection_id,
        name=collection.name,
        document_ids=MEMBER_IDS,
        expected_revision=1,
    )
    for index, decision, reason in (
        (0, "include", "Current synthetic human include."),
        (1, "exclude", REASON),
        (2, "exclude", REASON),
        (3, "exclude", REASON.lower()),
        (4, "exclude", " " + REASON),
        (5, "unsure", "A person still needs to decide."),
    ):
        container.paper_screening.submit(
            collection.collection_id,
            MEMBER_IDS[index],
            ScreeningSubmission.model_validate(
                {
                    "collection_revision": 2,
                    "expected_decision_revision": 0,
                    "decision": decision,
                    "reason": reason,
                }
            ),
        )
    return collection


def run_demo(output_dir: Path) -> str:
    """Save real response bytes immediately; setup writes use only a new temporary database."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    artifacts: dict[str, bytes] = {}
    counts: dict[str, int] = {}

    def save(name: str, content: bytes) -> None:
        with (output_dir / name).open("xb") as output:
            output.write(content)
        artifacts[name] = content

    with TemporaryDirectory(prefix="scholar-screening-report-") as temporary, ExitStack() as stack:
        network = {
            method: stack.enter_context(patch.object(target, method, side_effect=_no_work))
            for target, method in ((socket, "create_connection"), (socket.socket, "connect"))
        }
        database = Path(temporary) / "synthetic.sqlite3"
        settings = offline_settings(database)
        app = create_app(settings)
        container: AppContainer = app.state.container
        collection = _seed(container)
        before = database.read_bytes()
        with (
            TestClient(app) as client,
            browsing_guards(container) as guards,
            ExitStack() as reads,
        ):
            extra = {
                method: reads.enter_context(patch.object(target, method, side_effect=_no_work))
                for target, method in (
                    (container.paper_screening, "submit"),
                    (container.paper_screening, "list_queue"),
                    (container.paper_screening, "list_history"),
                    (container.event_log, "append_transition"),
                )
            }

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
                    raise RuntimeError("A response lost its privacy header.")
                save(name, response.content)
                return SearchPage(response.text)

            catalog = get("catalog.html", "/explore")
            listed = get("collections.html", catalog.links["saved-collections"])
            selected = urlsplit(listed.links["inspect-collection-1"])
            detail = get(
                "collection.html",
                selected.path
                + "?"
                + urlencode(
                    {**dict(parse_qsl(selected.query)), "limit": 1, "status": "unscreened"}
                ),
            )
            if detail.documents != [MEMBER_IDS[-1]]:
                raise RuntimeError("The visible unscreened selection was not preserved.")
            for format, extension in (("json", "json"), ("markdown", "md")):
                get(f"report.{extension}", detail.links["download-screening-report-" + format])
                direct = export_screening_report(
                    container.paper_screening,
                    collection.collection_id,
                    collection_revision=2,
                    format=format,
                )
                if direct.content != artifacts[f"report.{extension}"]:
                    raise RuntimeError("Python and API report bytes differ.")
            report = ScreeningReport.model_validate_json(artifacts["report.json"])
            source = (report.snapshot.model_dump_json(indent=2) + "\n").encode("utf-8")
            save("snapshot.json", source)
            if report.snapshot_sha256 != hashlib.sha256(source).hexdigest():
                raise RuntimeError("Report digest does not identify its complete source export.")
            stale = client.get(
                f"/collections/{collection.collection_id}/screening-report",
                params={"collection_revision": 1},
            )
            save("stale-revision.json", stale.content)
            counts.update({name: guard.call_count for name, guard in guards.items()})
            counts.update({name: guard.call_count for name, guard in extra.items()})
        restarted = create_app(settings)
        with TestClient(restarted) as client, browsing_guards(restarted.state.container) as guards:
            equal = []
            for format, extension in (("json", "json"), ("markdown", "md")):
                response = client.get(
                    f"/collections/{collection.collection_id}/screening-report?"
                    + urlencode({"collection_revision": 2, "format": format})
                )
                response.raise_for_status()
                equal.append(response.content == artifacts[f"report.{extension}"])
            counts.update({f"restart:{name}": guard.call_count for name, guard in guards.items()})
        counts.update({f"socket:{name}": guard.call_count for name, guard in network.items()})
        checks = {
            "synthetic_only": True,
            "collection_id": collection.collection_id,
            "collection_revision": report.snapshot.collection_revision,
            "visible_members": len(detail.documents),
            "report_members": report.snapshot.total_documents,
            "counts": report.snapshot.counts.model_dump(),
            "resolved": report.resolved_documents,
            "needs_review": report.needs_review_documents,
            "exclusion_group_counts": [group.count for group in report.exclusion_reasons],
            "snapshot_sha256": report.snapshot_sha256,
            "stale_revision_http_status": stale.status_code,
            "restart_identical": all(equal),
            "database_bytes_unchanged": database.read_bytes() == before,
            "event_count": len(container.event_log.list_events()),
            "forbidden_call_counts": counts,
        }
        if (
            any(counts.values())
            or checks["event_count"] != 0
            or not checks["database_bytes_unchanged"]
            or not checks["restart_identical"]
            or stale.status_code != 409
            or report.snapshot.total_documents != 9
            or report.resolved_document_ids != MEMBER_IDS[:5]
            or report.needs_review_document_ids != MEMBER_IDS[5:]
            or checks["exclusion_group_counts"] != [2, 1, 1]
        ):
            raise RuntimeError("Measured complete/read-only screening report checks failed.")
    portable = ScreeningReport.model_validate_json((output_dir / "report.json").read_bytes())
    checks["database_removed"] = not database.exists()
    checks["offline_json_reproduces_markdown"] = (
        portable.to_markdown().encode("utf-8") == artifacts["report.md"]
    )
    if not checks["database_removed"] or not checks["offline_json_reproduces_markdown"]:
        raise RuntimeError("The portable report did not reproduce after removing its database.")
    panels = [
        f"{PANEL_TITLES[0]}\n"
        "Executed: corpus explorer -> saved collections -> human progress download.\n"
        f"Collection revision: {report.snapshot.collection_revision}; "
        f"visible filtered members: {checks['visible_members']}.\n"
        f"Complete report members: {report.snapshot.total_documents}; no hidden duplicates.\n"
        "Original synthetic notes and human decisions, not a real literature review.",
        f"{PANEL_TITLES[1]}\n"
        + "; ".join(
            f"{status}: {count}" for status, count in report.snapshot.counts.model_dump().items()
        )
        + ".\n"
        f"Resolved: {report.resolved_documents}; "
        f"still needing human review: {report.needs_review_documents}.\n"
        "Resolved means current include or exclude, not scientific validation.\n"
        "Both earlier include and exclude opinions are stale, not active decisions.",
        f"{PANEL_TITLES[2]}\n"
        f"Current excluded members: {report.snapshot.counts.exclude}.\n"
        f"Exact reason group counts: {checks['exclusion_group_counts']}.\n"
        "Case and leading whitespace stay distinct; identical reasons group together.\n"
        "JSON and literal Markdown retain exact member IDs and original multiline reasons.\n"
        "No PRISMA stages, automatic labels, recall estimates or stopping claims.",
        f"{PANEL_TITLES[3]}\n"
        f"Actual JSON: {len(artifacts['report.json'])} bytes; "
        f"Markdown: {len(artifacts['report.md'])} bytes.\n"
        f"Stale revision: HTTP {stale.status_code}; restart bytes identical: {all(equal)}.\n"
        f"Database unchanged: {checks['database_bytes_unchanged']}; "
        f"events: {checks['event_count']}.\n"
        f"Forbidden work calls: {sum(counts.values())}; portable Markdown reproduced offline.\n"
        "Executed synthetic output illustration, not browser footage or a benchmark.",
    ]
    transcript = "\n\n".join(panels) + "\n"
    save("transcript.txt", transcript.encode("utf-8"))
    checks["output_bytes"] = {name: len(content) for name, content in artifacts.items()}
    checks["sha256"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()
    }
    save("checks.json", (json.dumps(checks, indent=2) + "\n").encode("utf-8"))
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(run_demo(args.output_dir), end="")


if __name__ == "__main__":
    main()
