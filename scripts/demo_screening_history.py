"""Execute synthetic local history reads; illustrate actual output, never scientific results."""

import argparse
import json
import sqlite3
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlencode

from fastapi.testclient import TestClient

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from scripts.demo_collection_browser import _extra_guards
from scripts.demo_corpus_explorer import browsing_guards
from scripts.demo_evidence_export import offline_settings
from scripts.demo_passage_search import SearchPage
from storage.paper_screening import ScreeningHistory, ScreeningReview

SELECTED_ID = "synthetic/paper-\u7814?draft#v1"
PANEL_TITLES = (
    "1. Keep a legacy boundary, not invented history",
    "2. Retain each successful human decision",
    "3. Browse older revisions without selecting evidence",
    "4. Restart and measure read-only behavior",
)


def run_demo(output_dir: Path) -> str:
    """Persist actual JSON/HTML and measured checks from an isolated temporary database."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a new directory.")
    output_dir.mkdir(parents=True)
    artifacts: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    with TemporaryDirectory(prefix="screening-history-") as temporary:
        database = Path(temporary) / "corpus.sqlite3"
        settings = offline_settings(database)
        app = create_app(settings)
        container: AppContainer = app.state.container
        container.document_store.add_documents(
            [
                Document(
                    document_id=identifier,
                    title="Original synthetic history fixture",
                    text="Not a publication or a scientific finding.",
                    source="synthetic:screening-history",
                )
                for identifier in (SELECTED_ID, "legacy")
            ],
            [],
        )
        collection = container.paper_collections.create(
            name="Synthetic human screening", document_ids=[SELECTED_ID, "legacy"]
        )
        # A declared latest-only fixture models pre-upgrade storage; no history is fabricated.
        legacy = ScreeningReview(
            schema_version="1.0",
            collection_id=collection.collection_id,
            document_id="legacy",
            collection_revision=1,
            revision=1,
            decision="include",
            reason="Synthetic legacy latest-only opinion.",
            updated_at=datetime.now(UTC),
        )
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute(
                "INSERT INTO paper_screening_decisions (collection_id, document_id, record) "
                "VALUES (?, ?, ?)",
                (collection.collection_id, "legacy", legacy.model_dump_json()),
            )
        base = f"/collections/{collection.collection_id}/screening"
        legacy_page = container.paper_screening.list_history(
            collection.collection_id, "legacy", collection_revision=1
        )
        artifacts["legacy-before.json"] = legacy_page.model_dump_json(indent=2).encode()
        saved: list[ScreeningReview] = []
        with TestClient(app) as client:
            for revision, decision in enumerate(("include", "exclude", "unsure")):
                response = client.put(
                    f"{base}/{quote(SELECTED_ID, safe='')}",
                    json={
                        "collection_revision": 1,
                        "expected_decision_revision": revision,
                        "decision": decision,
                        "reason": (
                            f"Synthetic human {decision} reason; <script>is literal</script>."
                        ),
                    },
                )
                response.raise_for_status()
                saved.append(ScreeningReview.model_validate_json(response.content))
            retry = client.put(
                f"{base}/{quote(SELECTED_ID, safe='')}",
                json={
                    "collection_revision": 1,
                    "expected_decision_revision": 0,
                    "decision": "include",
                    "reason": "Rejected stale synthetic retry.",
                },
            )
            resumed = client.put(
                f"{base}/legacy",
                json={
                    "collection_revision": 1,
                    "expected_decision_revision": 1,
                    "decision": "exclude",
                    "reason": "First retained synthetic legacy update.",
                },
            )
            resumed.raise_for_status()
        collection = container.paper_collections.replace(
            collection.collection_id,
            name=collection.name,
            document_ids=collection.document_ids,
            expected_revision=1,
        )
        before = database.read_bytes()
        with (
            TestClient(app) as client,
            browsing_guards(container) as guards,
            ExitStack() as stack,
        ):
            extra = _extra_guards(stack, container)

            def get(filename: str, url: str) -> bytes:
                response = client.get(url)
                response.raise_for_status()
                if response.headers["cache-control"] != "no-store":
                    raise RuntimeError("A history response lost its privacy header.")
                artifacts[filename] = response.content
                return response.content

            detail = SearchPage(
                get(
                    "collection.html",
                    "/explore/collection?"
                    + urlencode(
                        {
                            "collection_id": collection.collection_id,
                            "collection_revision": 2,
                        }
                    ),
                ).decode()
            )
            history_url = detail.links["screening-history-1"] + "&limit=2"
            html = SearchPage(get("history.html", history_url).decode())
            page = ScreeningHistory.model_validate_json(
                get("history.json", html.links["history-json"])
            )
            older_html = SearchPage(get("older.html", html.links["next-page"]).decode())
            older = ScreeningHistory.model_validate_json(
                get("older.json", older_html.links["history-json"])
            )
            python_page = container.paper_screening.list_history(
                collection.collection_id, SELECTED_ID, collection_revision=2, limit=2
            )
            legacy_after = container.paper_screening.list_history(
                collection.collection_id, "legacy", collection_revision=2
            )
            artifacts["legacy-after.json"] = legacy_after.model_dump_json(indent=2).encode()
            conflict = client.get(
                base + "/history",
                params={
                    "document_id": SELECTED_ID,
                    "collection_revision": 1,
                },
            )
            counts.update({key: guard.call_count for key, guard in guards.items()})
            counts.update({f"extra-{index}": guard.call_count for index, guard in enumerate(extra)})
        restarted = create_app(settings)
        with (
            TestClient(restarted) as client,
            browsing_guards(restarted.state.container) as guards,
            ExitStack() as stack,
        ):
            extra = _extra_guards(stack, restarted.state.container)
            reopened = client.get(history_url)
            reopened.raise_for_status()
            artifacts["restarted.html"] = reopened.content
            counts.update({f"restart:{key}": guard.call_count for key, guard in guards.items()})
            counts.update(
                {f"restart:extra-{index}": guard.call_count for index, guard in enumerate(extra)}
            )
        checks = {
            "fixture": "Original synthetic notes and an explicitly seeded latest-only legacy row.",
            "legacy_before_items": len(legacy_page.items),
            "legacy_first_retained_revision": legacy_after.first_retained_revision,
            "legacy_unretained_decision_revisions": legacy_after.unretained_decision_revisions,
            "decisions_newest_first": [item.decision for item in page.items + older.items],
            "retained_decisions_equal_submissions": page.items + older.items
            == list(reversed(saved)),
            "first_page_revisions": [item.revision for item in page.items],
            "older_page_revisions": [item.revision for item in older.items],
            "current_status": page.current_status,
            "rejected_retry_status": retry.status_code,
            "stale_collection_status": conflict.status_code,
            "python_http_equal": python_page == page,
            "exact_identity_and_return": (
                html.documents == [SELECTED_ID]
                and html.links["back-to-collection"] == older_html.links["back-to-collection"]
            ),
            "restart_html_identical": artifacts["history.html"] == artifacts["restarted.html"],
            "database_bytes_unchanged_during_reads": database.read_bytes() == before,
            "event_count": len(container.event_log.list_events()),
            "forbidden_call_counts": counts,
            "output_bytes": {name: len(content) for name, content in artifacts.items()},
        }
        if (
            any(counts.values())
            or checks["event_count"] != 0
            or checks["legacy_before_items"] != 0
            or checks["legacy_first_retained_revision"] != 2
            or checks["legacy_unretained_decision_revisions"] != 1
            or checks["first_page_revisions"] != [3, 2]
            or checks["older_page_revisions"] != [1]
            or checks["current_status"] != "stale"
            or retry.status_code != 409
            or conflict.status_code != 409
            or not all(
                checks[key]
                for key in (
                    "retained_decisions_equal_submissions",
                    "python_http_equal",
                    "exact_identity_and_return",
                    "restart_html_identical",
                    "database_bytes_unchanged_during_reads",
                )
            )
        ):
            raise RuntimeError("Executed screening-history contract failed.")
    checks["temporary_database_removed"] = not database.exists()
    if not checks["temporary_database_removed"]:
        raise RuntimeError("Temporary synthetic database was not removed.")
    transcript = (
        "\n\n".join(
            (
                f"{PANEL_TITLES[0]}\n"
                f"Legacy latest-only fixture: {checks['legacy_before_items']} history rows "
                "before update.\n"
                f"First retained revision: {checks['legacy_first_retained_revision']}; "
                f"unretained prior revisions: {checks['legacy_unretained_decision_revisions']}.\n"
                "The earlier reason remains latest-only until updated; "
                "no old event was backfilled.",
                f"{PANEL_TITLES[1]}\n"
                "Executed include -> exclude -> unsure through the existing PUT API.\n"
                "Retained records equal submissions: "
                f"{checks['retained_decisions_equal_submissions']}.\n"
                f"Original reasons/timestamps preserved; stale retry: HTTP {retry.status_code}.\n"
                "Each successful revision is appended in the latest-decision transaction.",
                f"{PANEL_TITLES[2]}\n"
                f"Native history first page: {checks['first_page_revisions']}; "
                f"older page: {checks['older_page_revisions']}.\n"
                f"After collection edit, current status: {page.current_status}; "
                f"old link: HTTP {conflict.status_code}.\n"
                f"Python / HTTP equal: {checks['python_http_equal']}; "
                "exact identity and return preserved.\n"
                "Historical includes are opinions, not current evidence selection.",
                f"{PANEL_TITLES[3]}\n"
                f"Restart HTML identical: {checks['restart_html_identical']}.\n"
                "Database unchanged during reads: "
                f"{checks['database_bytes_unchanged_during_reads']}.\n"
                f"Events: {checks['event_count']}; forbidden calls: {sum(counts.values())}; "
                "fixture removed.\n"
                "Executed synthetic output illustration, not browser frames "
                "or a scientific benchmark.",
            )
        )
        + "\n"
    )
    artifacts["checks.json"] = (json.dumps(checks, ensure_ascii=True, indent=2) + "\n").encode()
    artifacts["transcript.txt"] = transcript.encode()
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)
    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(run_demo(parser.parse_args().output_dir), end="")


if __name__ == "__main__":
    main()
