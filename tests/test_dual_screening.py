"""Independent dual-reviewer screening: blinding, conflicts, resolutions and agreement."""

import csv
import io
import json
import socket
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from storage.document_store import SQLiteDocumentStore
from storage.dual_screening import MAX_NAMED_REVIEWERS, report_summary
from storage.paper_collections import CollectionError, PaperCollection
from storage.paper_screening import ScreeningSubmission, SQLitePaperScreening
from storage.screening_report import export_screening_report

IDS = (
    "doi:10.5555/dual-0",
    "paper/../one?draft#1",
    "paper-two",
    "paper-three",
    "paper-four",
    "paper-five",
)
# r1 rows / r2 columns: [[1, 1, 0], [1, 2, 0], [0, 1, 0]] on include/exclude/unsure.
R1 = ("include", "include", "exclude", "exclude", "exclude", "unsure")
R2 = ("include", "exclude", "exclude", "exclude", "include", "exclude")
TEAM_ROUTES = ("screening-conflicts", "screening-agreement", "screening-final-decisions")


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Independent screening must not use the network.")


@dataclass
class Team:
    app: FastAPI
    client: TestClient
    container: AppContainer
    path: Path
    collection: PaperCollection

    @property
    def base(self) -> str:
        return f"/collections/{self.collection.collection_id}"

    def decide(
        self,
        reviewer: str,
        index: int,
        decision: str,
        *,
        reason: str | None = None,
        expected: int = 0,
        revision: int = 1,
    ) -> httpx.Response:
        return self.client.put(
            f"{self.base}/reviewer-screening/{quote(IDS[index], safe='')}",
            json={
                "reviewer": reviewer,
                "collection_revision": revision,
                "expected_decision_revision": expected,
                "decision": decision,
                "reason": reason or f"{reviewer.upper()}_PRIVATE reason {index}",
            },
        )

    def get(self, route: str, **params: object) -> httpx.Response:
        return self.client.get(f"{self.base}/{route}", params={"collection_revision": 1, **params})

    def protocol(self, *, blind: bool, must_match: bool = False, expected: int) -> dict[str, Any]:
        response = self.client.put(
            f"{self.base}/screening-protocol",
            json={
                "blind": blind,
                "exclusion_reasons_must_match": must_match,
                "expected_protocol_revision": expected,
                "changed_by": "lead",
            },
        )
        return ok(response)


def ok(response: httpx.Response) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    result: dict[str, Any] = response.json()
    return result


def failed(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"
    assert "PRIVATE" not in response.text


def seed(path: Path) -> None:
    SQLiteDocumentStore(path).add_documents(
        [
            Document(
                document_id=identifier,
                title=f"Synthetic dual-review paper {index}",
                text=f"Synthetic grounded retrieval abstract {index} with cited evidence.",
                source="synthetic:dual-review",
            )
            for index, identifier in enumerate(IDS)
        ],
        [],
    )


@pytest.fixture
def team(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Team]:
    for target, method in (
        (httpx.HTTPTransport, "handle_request"),
        (httpx.AsyncHTTPTransport, "handle_async_request"),
        (socket, "create_connection"),
        (socket.socket, "connect"),
    ):
        monkeypatch.setattr(target, method, denied)
    path = tmp_path / "team.sqlite3"
    app = create_app(offline_settings(path))
    seed(path)
    container: AppContainer = app.state.container
    collection = container.paper_collections.create(name="Synthetic dual review", document_ids=IDS)
    with TestClient(app) as client:
        yield Team(app, client, container, path, collection)


def screen_all(team: Team) -> None:
    for reviewer, decisions in (("r1", R1), ("r2", R2)):
        for index, decision in enumerate(decisions):
            assert team.decide(reviewer, index, decision).status_code == 200


def test_blind_mode_hides_other_reviewers_on_every_reviewer_surface(team: Team) -> None:
    legacy = {
        route: team.get(route).content
        for route in ("screening", "screening/export", "screening-priority")
    }
    for index in (0, 1):
        ok(team.decide("r2", index, R2[index]))
    before = team.get("reviewer-screening-priority", reviewer="r2").content
    for index, decision in enumerate(R1):
        recorded = ok(team.decide("r1", index, decision))
        assert recorded["reviewer"] == "r1" and recorded["blinded"] is True
        assert "R2_PRIVATE" not in json.dumps(recorded)
    queue = team.get("reviewer-screening", reviewer="r2")
    items = ok(queue)["items"]
    assert [item["others_hidden"] for item in items] == [False, False, True, True, True, True]
    assert all(item["other_decisions"] is None and item["final"] is None for item in items[2:])
    assert items[0]["other_decisions"][0]["reviewer"] == "r1"
    assert items[1]["final"]["status"] == "conflict"
    assert "R1_PRIVATE reason 1" in queue.text
    assert not any(f"R1_PRIVATE reason {index}" in queue.text for index in range(2, 6))
    assert ok(queue)["counts"] == {
        "unscreened": 4,
        "include": 1,
        "exclude": 1,
        "unsure": 0,
        "stale": 0,
    }
    hidden_history = team.get("reviewer-screening-history", reviewer="r2", document_id=IDS[3])
    assert ok(hidden_history)["items"] == [] and "R1" not in hidden_history.text
    after = team.get("reviewer-screening-priority", reviewer="r2")
    assert after.content == before and ok(after)["priority"]["model_status"] == "ready"
    failed(
        team.get("reviewer-screening", reviewer="r2", blind="false"), 409, "screening_blind_mode_on"
    )
    for route in TEAM_ROUTES:
        failed(team.get(route), 409, "screening_blind_mode_on")
    report = ok(team.get("screening-report"))["dual_review"]
    assert report["hidden"] is True and report["final_counts"] is None and report["pairs"] is None
    assert [entry["current_decisions"] for entry in report["reviewers"]] == [6, 2]
    markdown = team.get("screening-report", format="markdown").text
    assert "Blind mode is on" in markdown and "PRIVATE" not in markdown
    for route, content in legacy.items():
        assert team.get(route).content == content


def test_blinding_lifts_per_paper_and_returns_after_a_collection_edit(team: Team) -> None:
    ok(team.decide("r1", 0, "include"))
    ok(team.decide("r2", 0, "exclude"))
    member = ok(team.get("reviewer-screening", reviewer="r2"))["items"][0]
    assert member["other_decisions"][0]["decision"] == "include"
    renamed = team.client.put(
        team.base,
        json={"name": "Revised criteria", "document_ids": list(IDS), "expected_revision": 1},
    )
    assert renamed.status_code == 200
    stale = ok(team.get("reviewer-screening", reviewer="r2", collection_revision=2))["items"][0]
    assert stale["status"] == "stale" and stale["others_hidden"] is True
    redecided = ok(team.decide("r2", 0, "include", expected=1, revision=2))
    assert redecided["blinded"] is False  # r2 could have seen r1's earlier decision
    lifted = ok(team.get("reviewer-screening", reviewer="r2", collection_revision=2))["items"][0]
    assert lifted["other_decisions"][0]["current"] is False
    assert lifted["final"]["status"] == "single_review"
    team.protocol(blind=False, expected=0)
    first = ok(team.decide("r3", 1, "exclude", revision=2))
    assert first["blinded"] is False
    open_view = ok(team.get("reviewer-screening", reviewer="r3", collection_revision=2))
    assert open_view["blind"] is False and not open_view["items"][0]["others_hidden"]
    kept_blind = ok(
        team.get("reviewer-screening", reviewer="r3", collection_revision=2, blind="true")
    )
    assert kept_blind["blind"] is True and kept_blind["collection_blind"] is False
    assert kept_blind["items"][0]["others_hidden"] is True
    assert kept_blind["items"][1]["others_hidden"] is False


def test_resolutions_are_append_only_final_and_never_change_kappa(team: Team) -> None:
    screen_all(team)
    failed(
        team.client.put(
            f"{team.base}/screening-resolutions/{quote(IDS[1], safe='')}",
            json={
                "resolver": "lead",
                "collection_revision": 1,
                "expected_resolution_revision": 0,
                "expected_basis": [
                    {"reviewer": "r1", "revision": 1},
                    {"reviewer": "r2", "revision": 1},
                ],
                "decision": "exclude",
                "rationale": "Blind resolution attempt",
            },
        ),
        409,
        "screening_blind_mode_on",
    )
    team.protocol(blind=False, expected=0)
    agreement = ok(team.get("screening-agreement"))
    conflicts = ok(team.get("screening-conflicts"))
    assert [item["document_id"] for item in conflicts["items"]] == [IDS[1], IDS[4], IDS[5]]
    first_page = ok(team.get("screening-conflicts", limit=1))
    assert first_page["next_cursor"] == IDS[1] and first_page["counts"] == conflicts["counts"]
    second_page = ok(team.get("screening-conflicts", limit=1, cursor=IDS[1]))
    assert [item["document_id"] for item in second_page["items"]] == [IDS[4]]
    item = conflicts["items"][0]
    url = f"{team.base}/screening-resolutions/{quote(IDS[1], safe='')}"
    body = {
        "resolver": "lead",
        "collection_revision": 1,
        "expected_resolution_revision": item["resolution_revision"],
        "expected_basis": item["expected_basis"],
        "decision": "exclude",
        "rationale": "Discussed: wrong population",
    }
    resolution = ok(team.client.put(url, json=body))
    assert resolution["revision"] == 1 and resolution["conflict_type"] == "decision"
    assert [entry["decision"] for entry in resolution["basis"]] == ["include", "exclude"]
    failed(team.client.put(url, json=body), 409, "screening_resolution_revision_conflict")
    failed(
        team.client.put(url, json={**body, "expected_basis": item["expected_basis"][:1] * 2}),
        409,
        "screening_conflict_changed",
    )
    failed(
        team.client.put(
            f"{team.base}/screening-resolutions/{quote(IDS[0], safe='')}",
            json={**body, "expected_basis": item["expected_basis"]},
        ),
        409,
        "screening_no_conflict",
    )
    final = ok(team.get("screening-final-decisions"))
    resolved = final["items"][1]["final"]
    assert (resolved["status"], resolved["decision"], resolved["source"]) == (
        "resolved",
        "exclude",
        "resolution",
    )
    assert resolved["exclusion_reason"] == "Discussed: wrong population"
    assert [entry["decision"] for entry in final["items"][1]["decisions"]] == ["include", "exclude"]
    assert final["final_counts"]["resolved_exclude"] == 1 and final["final_counts"]["conflict"] == 2
    rows = list(
        csv.DictReader(io.StringIO(team.get("screening-final-decisions", format="csv").text))
    )
    resolved_rows = [row for row in rows if row["document_id"] == "'" + IDS[1]]
    assert [row["reviewer"] for row in resolved_rows] == ["'r1", "'r2"]
    assert {row["final_decision"] for row in resolved_rows} == {"'exclude"}
    assert {row["reviewer_decision"] for row in resolved_rows} == {"'include", "'exclude"}
    report = ok(team.get("screening-report"))
    assert report["snapshot"]["counts"]["unscreened"] == len(IDS)  # default reviewer unchanged
    assert report["dual_review"]["final_counts"]["resolved_exclude"] == 1
    assert report["dual_review"]["conflicts"] == {
        "disagreements": 3,
        "resolved": 1,
        "unresolved": 2,
        "decision": 3,
        "exclusion_reason": 0,
        "outdated_resolutions": 0,
    }
    later = ok(team.get("screening-agreement"))
    assert later["pairs"] == agreement["pairs"]
    corrected = ok(
        team.client.put(
            url, json={**body, "expected_resolution_revision": 1, "decision": "include"}
        )
    )
    assert corrected["revision"] == 2
    with sqlite3.connect(team.path) as connection:
        assert connection.execute(
            "SELECT revision FROM paper_screening_resolutions ORDER BY revision"
        ).fetchall() == [(1,), (2,)]
    ok(team.decide("r2", 1, "include", expected=1))
    outdated = ok(team.get("screening-final-decisions"))["items"][1]["final"]
    assert outdated["status"] == "agreed" and outdated["resolution_status"] == "outdated"
    assert ok(team.get("screening-conflicts"))["counts"]["outdated_resolutions"] == 1


def test_exclusion_reason_conflicts_follow_the_protocol(team: Team) -> None:
    for reviewer, reason in (("r1", "Wrong population"), ("r2", "Wrong outcome")):
        ok(team.decide(reviewer, 0, "exclude", reason=reason))
        ok(team.decide(reviewer, 1, "exclude", reason="Wrong design"))
    team.protocol(blind=False, expected=0)
    final = ok(team.get("screening-final-decisions"))["items"]
    assert final[0]["final"]["status"] == "agreed"
    assert final[0]["final"]["exclusion_reasons_differ"] is True
    assert final[0]["final"]["exclusion_reason"] is None
    assert final[1]["final"]["exclusion_reason"] == "Wrong design"
    assert ok(team.get("screening-conflicts"))["items"] == []
    agreement = ok(team.get("screening-agreement"))
    assert agreement["final_excludes_without_single_reason"] == [IDS[0]]
    team.protocol(blind=False, must_match=True, expected=1)
    stale = {"blind": True, "exclusion_reasons_must_match": True, "changed_by": "lead"}
    failed(
        team.client.put(
            f"{team.base}/screening-protocol", json={**stale, "expected_protocol_revision": 1}
        ),
        409,
        "screening_protocol_revision_conflict",
    )
    failed(
        team.client.put(
            f"{team.base}/screening-protocol",
            json={**stale, "blind": False, "expected_protocol_revision": 2},
        ),
        422,
        "screening_protocol_unchanged",
    )
    conflicts = ok(team.get("screening-conflicts"))["items"]
    assert [(item["document_id"], item["conflict_type"]) for item in conflicts] == [
        (IDS[0], "exclusion_reason")
    ]
    stats = ok(team.get("screening-agreement"))["pairs"][0]["statistics"]
    assert stats["agreements"] == 2 and stats["kappa_undefined_reason"] == "chance_agreement_is_one"


def test_agreement_report_formats_carry_hand_computed_statistics(team: Team) -> None:
    screen_all(team)
    team.protocol(blind=False, expected=0)
    response = team.get("screening-agreement")
    report = ok(response)
    assert response.headers["content-disposition"] == (
        'attachment; filename="screening-agreement-Synthetic-dual-review-'
        f'{team.collection.collection_id}-r1.json"'
    )
    primary, binary = report["pairs"]
    assert (primary["reviewer_a"], primary["reviewer_b"]) == ("r1", "r2")
    # p_o = 3/6; p_e = (2*2 + 3*4 + 1*0) / 36 = 16/36; kappa = (1/2 - 4/9) / (5/9) = 0.1.
    assert primary["statistics"]["confusion"] == [[1, 1, 0], [1, 2, 0], [0, 1, 0]]
    assert primary["statistics"]["kappa"] == 0.1
    assert primary["statistics"]["pabak"] == 0.25
    assert primary["statistics"]["landis_koch_band"] == "slight"
    # Unsure counted as include: p_o = 3/6, p_e = (3*2 + 3*4) / 36 = 1/2, kappa = 0.
    assert binary["statistics"]["confusion"] == [[1, 2], [1, 2]]
    assert binary["statistics"]["kappa"] == 0.0 and binary["statistics"]["pabak"] == 0.0
    assert primary["unblinded_decisions"] == 0 and primary["unknown_blinding_decisions"] == 0
    assert report["conflicts"]["unresolved"] == 3
    assert report["unresolved_conflict_document_ids"] == [IDS[1], IDS[4], IDS[5]]
    assert report["final_counts"]["double_screened"] == 6
    assert "Fleiss, Cohen and Everitt (1969)" in report["method"]["standard_error"]
    table = team.get("screening-agreement", format="csv")
    assert table.headers["content-type"] == "text/csv; charset=utf-8"
    rows = list(csv.DictReader(io.StringIO(table.text)))
    assert [(row["scale"], row["kappa"], row["pabak"]) for row in rows] == [
        ("'include_exclude_unsure", "0.1", "0.25"),
        ("'include_or_unsure_vs_exclude", "0.0", "0.0"),
    ]
    assert rows[0]["csv_text_encoding"] == "'apostrophe-prefix-v1"
    assert rows[0]["confusion"] == "'[[1, 1, 0], [1, 2, 0], [0, 1, 0]]"
    markdown = team.get("screening-agreement", format="markdown").text
    assert markdown.startswith("# Independent screening agreement report\n")
    assert "| r1 / r2 | include_exclude_unsure | 6 | 0.500 | 0.444 | 0.100 |" in markdown
    assert "| unsure | 0 | 1 | 0 |" in markdown and "arbitrary" in markdown


def test_unlabeled_decisions_are_the_default_reviewer_without_rewrites(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    seed(path)
    legacy = SQLitePaperScreening(path)
    collection = legacy.collections.create(name="Legacy review", document_ids=IDS)
    for index, decision in enumerate(R1[:3]):
        legacy.submit(
            collection.collection_id,
            IDS[index],
            ScreeningSubmission(
                collection_revision=1,
                expected_decision_revision=0,
                decision=decision,
                reason=f"Legacy unlabeled reason {index}",
            ),
        )
    export = legacy.export_results(collection.collection_id, collection_revision=1).content
    report = export_screening_report(legacy, collection.collection_id, collection_revision=1)
    assert (
        report_summary(legacy, collection.collection_id, collection_revision=1, expected_default={})
        is None
    )

    def rows() -> list[tuple[Any, ...]]:
        with sqlite3.connect(path) as connection:
            return [
                *connection.execute("SELECT * FROM paper_screening_decisions ORDER BY 1, 2"),
                *connection.execute("SELECT * FROM paper_screening_history ORDER BY 1, 2, 3"),
            ]

    original = rows()
    app = create_app(offline_settings(path))
    with TestClient(app) as client:
        base = f"/collections/{collection.collection_id}"
        assert rows() == original
        params = {"collection_revision": 1}
        assert client.get(f"{base}/screening/export", params=params).content == export
        assert client.get(f"{base}/screening-report", params=params).content == report.content
        for index, decision in enumerate(R2[:3]):
            response = client.put(
                f"{base}/reviewer-screening/{quote(IDS[index], safe='')}",
                json={
                    "reviewer": "r2",
                    "collection_revision": 1,
                    "expected_decision_revision": 0,
                    "decision": decision,
                    "reason": "Second independent reviewer",
                },
            )
            assert response.status_code == 200
        assert rows() == original
        assert client.get(f"{base}/screening/export", params=params).content == export
        hidden = ok(client.get(f"{base}/reviewer-screening", params={**params, "reviewer": "r2"}))
        assert [item["others_hidden"] for item in hidden["items"][:3]] == [False] * 3
        assert hidden["items"][0]["other_decisions"][0]["reviewer"] == "default"
        assert hidden["items"][0]["other_decisions"][0]["blinded"] is None
        ok(
            client.put(
                f"{base}/screening-protocol",
                json={
                    "blind": False,
                    "exclusion_reasons_must_match": False,
                    "expected_protocol_revision": 0,
                    "changed_by": "lead",
                },
            )
        )
        agreement = ok(client.get(f"{base}/screening-agreement", params=params))
        pair = agreement["pairs"][0]
        assert (pair["reviewer_a"], pair["reviewer_b"]) == ("default", "r2")
        assert pair["unknown_blinding_decisions"] == 3 and pair["unblinded_decisions"] == 0
        conflicts = ok(client.get(f"{base}/screening-conflicts", params=params))["items"]
        assert [item["document_id"] for item in conflicts] == [IDS[1]]
        with pytest.raises(CollectionError) as changed:
            report_summary(
                app.state.container.paper_screening,
                collection.collection_id,
                collection_revision=1,
                expected_default=dict.fromkeys(IDS),
            )
        assert changed.value.code == "screening_report_snapshot_changed"
        updated = json.loads(client.get(f"{base}/screening-report", params=params).content)
        assert updated["snapshot"] == json.loads(report.content)["snapshot"]
        assert updated["dual_review"]["reviewers"][0] == {
            "reviewer": "default",
            "labeled": False,
            "current_decisions": 3,
            "stale_decisions": 0,
        }


@pytest.mark.parametrize(
    "label", ["default", "Alice", "a b", "a@b.example", "-a", "a-", "x" * 33, "", "é", "r\x00"]
)
def test_invalid_reviewer_labels_write_nothing(team: Team, label: str) -> None:
    failed(team.decide(label, 0, "include"), 422, "invalid_reviewer_screening_request")
    failed(
        team.get("reviewer-screening", reviewer=label), 422, "invalid_reviewer_screening_request"
    )
    with sqlite3.connect(team.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM paper_screening_reviewer_decisions"
        ).fetchone() == (0,)


@pytest.mark.parametrize(
    "query",
    [
        "collection_revision=1&reviewer=r1&unknown=1",
        "collection_revision=1&reviewer=r1&reviewer=r2",
        "collection_revision=+1&reviewer=r1",
        "collection_revision=1.0&reviewer=r1",
        "collection_revision=1&reviewer=r1&blind=yes",
        "collection_revision=1&reviewer=r1&limit=0",
        "reviewer=r1",
    ],
)
def test_query_validation_is_strict_and_echoes_nothing(team: Team, query: str) -> None:
    response = team.client.get(f"{team.base}/reviewer-screening?{query}")
    failed(response, 422, "invalid_reviewer_screening_request")
    assert "unknown" not in response.text


def test_methods_revisions_members_history_pages_and_reviewer_limit(team: Team) -> None:
    for route, allow in (("screening-protocol", "GET, PUT"), ("screening-conflicts", "GET")):
        response = team.client.post(f"{team.base}/{route}")
        assert response.status_code == 405 and response.headers["allow"] == allow
    response = team.client.get(f"{team.base}/reviewer-screening/{IDS[2]}")
    assert response.status_code == 405 and response.headers["allow"] == "PUT"
    for expected, decision in enumerate(("include", "exclude", "unsure")):
        ok(team.decide("r1", 2, decision, expected=expected))
    failed(team.decide("r1", 2, "include", expected=1), 409, "screening_revision_conflict")
    missing = team.client.put(
        f"{team.base}/reviewer-screening/not-a-member",
        json={
            "reviewer": "r1",
            "collection_revision": 1,
            "expected_decision_revision": 0,
            "decision": "include",
            "reason": "Not a member",
        },
    )
    failed(missing, 422, "screening_document_not_member")
    first = ok(team.get("reviewer-screening-history", reviewer="r1", document_id=IDS[2], limit=2))
    assert [item["revision"] for item in first["items"]] == [3, 2] and first["next_cursor"] == 2
    second = ok(
        team.get("reviewer-screening-history", reviewer="r1", document_id=IDS[2], limit=2, cursor=2)
    )
    assert [item["revision"] for item in second["items"]] == [1] and second["next_cursor"] is None
    failed(
        team.get("reviewer-screening-history", reviewer="r1", document_id=IDS[2], cursor=4),
        422,
        "screening_history_cursor_invalid",
    )
    for number in range(2, MAX_NAMED_REVIEWERS + 1):
        ok(team.decide(f"r{number}", 0, "include"))
    failed(team.decide("r9", 0, "include"), 409, "screening_reviewer_limit")
    ok(team.decide("r8", 1, "exclude"))
    failed(
        team.get("reviewer-screening", reviewer="r1", cursor="not-a-member"),
        422,
        "screening_cursor_invalid",
    )


def test_corrupt_saved_records_fail_closed_without_repair(team: Team) -> None:
    screen_all(team)
    team.protocol(blind=False, expected=0)
    item = ok(team.get("screening-conflicts"))["items"][0]
    ok(
        team.client.put(
            f"{team.base}/screening-resolutions/{quote(item['document_id'], safe='')}",
            json={
                "resolver": "lead",
                "collection_revision": 1,
                "expected_resolution_revision": 0,
                "expected_basis": item["expected_basis"],
                "decision": "include",
                "rationale": "Resolved for the corruption test",
            },
        )
    )
    for table, code, route in (
        ("paper_screening_resolutions", "invalid_screening_resolution_record", "conflicts"),
        ("paper_screening_reviewer_decisions", "invalid_reviewer_screening_record", "queue"),
        ("paper_screening_protocols", "invalid_screening_protocol_record", "protocol"),
    ):
        with sqlite3.connect(team.path) as connection:
            connection.execute(f"UPDATE {table} SET record = 'PRIVATE_CORRUPT'")  # noqa: S608
        response = {
            "conflicts": lambda: team.get("screening-conflicts"),
            "queue": lambda: team.get("reviewer-screening", reviewer="r1"),
            "protocol": lambda: team.client.get(f"{team.base}/screening-protocol"),
        }[route]()
        failed(response, 409, code)
    with sqlite3.connect(team.path) as connection:
        connection.execute("DELETE FROM paper_screening_protocols")
        connection.execute("DROP TABLE paper_screening_reviewer_decisions")
    failed(team.get("reviewer-screening", reviewer="r1"), 503, "collection_storage_error")
