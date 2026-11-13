"""Native reviewer and team pages: blind by default, explicit same-origin writes only."""

import json
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from storage.dual_screening import ReviewerSubmission
from tests.test_collection_explorer import detail_url
from tests.test_corpus_explorer import ExplorerDOM, assert_headers, href, values
from tests.test_dual_screening import IDS, R1, R2, Team
from tests.test_dual_screening import team as team
from tests.test_evidence_html import Node

REVIEWER = "/explore/reviewer-screening"
TEAM = "/explore/screening-conflicts"
PROTOCOL = "/explore/screening-protocol"
ORIGIN = {"origin": "http://testserver", "sec-fetch-site": "same-origin"}


def page(team: Team, path: str, **params: str | int) -> Node:
    response = team.client.get(path, params=params) if params else team.client.get(path)
    assert response.status_code == 200, response.text[:1000]
    assert_headers(response, referrer_policy="same-origin")
    assert "INDEPENDENT DUAL REVIEW" in response.text
    return ExplorerDOM(response.text).root


def form_fields(root: Node, identifier: str) -> dict[str, str]:
    form = root.by_id(identifier)
    assert form.attrs["method"] == "post"
    return {node.attrs["name"]: node.attrs.get("value", "") for node in form.by_tag("input")}


def seed_r1(team: Team) -> None:
    for index, decision in enumerate(R1):
        team.container.dual_screening.submit(
            team.collection.collection_id,
            IDS[index],
            ReviewerSubmission(
                reviewer="r1",
                collection_revision=1,
                expected_decision_revision=0,
                decision=decision,
                reason=f"R1_PRIVATE reason {index}",
            ),
        )


def opened(team: Team, **params: str | int) -> dict[str, str | int]:
    return {
        "collection_id": team.collection.collection_id,
        "collection_revision": 1,
        **params,
    }


def test_collection_pages_link_to_reviewer_and_team_views(team: Team) -> None:
    detail = team.client.get(detail_url(team.collection))
    assert detail.status_code == 200
    root = ExplorerDOM(detail.text).root
    entry = page(team, href(root, "reviewer-screening"))
    form = entry.by_id("reviewer-label-form")
    assert form.attrs["method"] == "get" and form.attrs["action"] == REVIEWER
    names = {node.attrs["name"] for node in form.by_tag("input")}
    assert names == {
        "reviewer",
        "collection_id",
        "collection_revision",
        "collection_return",
        "blind",
    }
    assert entry.by_id("reviewer").attrs["maxlength"] == "32"
    assert "checked" not in entry.by_id("blind").attrs
    assert href(entry, "back-to-collection").startswith("/explore/collection?")
    assert urlsplit(href(root, "screening-conflicts")).path == TEAM
    listed = ExplorerDOM(team.client.get("/explore/collections").text).root
    assert urlsplit(href(listed, "reviewer-screening-1")).path == REVIEWER


def test_reviewer_pages_stay_blind_until_your_own_decision(team: Team) -> None:
    seed_r1(team)
    queue_response = team.client.get(REVIEWER, params=opened(team, reviewer="r2"))
    assert "R1_PRIVATE" not in queue_response.text
    queue = page(team, REVIEWER, **opened(team, reviewer="r2"))
    assert len(queue.by_class("others-hidden")) == len(IDS)
    assert values(queue, "own-status") == ["unscreened"] * len(IDS)
    ordered = team.client.get(href(queue, "reviewer-priority"))
    assert ordered.status_code == 200 and "R1_PRIVATE" not in ordered.text
    member_url = href(queue, "open-member-2")
    member = team.client.get(member_url)
    assert member.status_code == 200 and "R1_PRIVATE" not in member.text
    root = ExplorerDOM(member.text).root
    assert root.by_id("others-hidden") is not None
    assert root.by_tag("textarea")[0].text().strip() == ""
    assert not any("selected" in option.attrs for option in root.by_tag("option")[1:])
    fields = form_fields(root, "reviewer-form")
    assert json.loads(fields["selection"])["reviewer"] == "r2"
    assert fields["expected_decision_revision"] == "0"
    reason = "Second reviewer <script>literal</script>\r\nreason"
    response = team.client.post(
        REVIEWER,
        data={**fields, "decision": "exclude", "reason": reason},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert response.status_code == 303
    target = urlsplit(response.headers["location"])
    assert target.path == REVIEWER and "reason" not in parse_qs(target.query)
    saved = page(team, response.headers["location"])
    assert saved.by_id("reviewer-saved").text().startswith("Your stored decision revision 1")
    assert values(saved, "own-review-reason") == [reason]
    assert values(saved, "other-review-reason") == ["R1_PRIVATE reason 1"]
    assert (
        "R1_PRIVATE reason 2"
        not in team.client.get(REVIEWER, params=opened(team, reviewer="r2")).text
    )
    stored = team.container.dual_screening.reviewer_history(
        team.collection.collection_id, IDS[1], reviewer="r2", collection_revision=1
    )
    assert stored.latest_review is not None and stored.latest_review.reason == reason
    assert stored.latest_review.blinded is True
    restarted = create_app(offline_settings(team.path))
    with TestClient(restarted) as client:
        assert (
            client.get(response.headers["location"]).content
            == team.client.get(response.headers["location"]).content
        )


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"origin": "http://evil.example"},
        {"origin": "null"},
        {**ORIGIN, "sec-fetch-site": "cross-site"},
    ],
)
def test_cross_origin_or_missing_origin_writes_nothing(team: Team, headers: dict[str, str]) -> None:
    root = page(team, REVIEWER, **opened(team, reviewer="r2", document_id=IDS[0]))
    response = team.client.post(
        REVIEWER,
        data={**form_fields(root, "reviewer-form"), "decision": "include", "reason": "Forged"},
        headers=headers,
        follow_redirects=False,
    )
    assert response.status_code == 403 and "location" not in response.headers
    queue = team.container.dual_screening.reviewer_queue(
        team.collection.collection_id, reviewer="r2", collection_revision=1
    )
    assert queue.counts.unscreened == len(IDS)


def test_team_page_reveals_nothing_until_an_explicit_protocol_change(team: Team) -> None:
    seed_r1(team)
    for index, decision in enumerate(R2):
        team.container.dual_screening.submit(
            team.collection.collection_id,
            IDS[index],
            ReviewerSubmission(
                reviewer="r2",
                collection_revision=1,
                expected_decision_revision=0,
                decision=decision,
                reason=f"R2_PRIVATE reason {index}",
            ),
        )
    blind = team.client.get(TEAM, params=opened(team))
    assert blind.status_code == 200 and "PRIVATE" not in blind.text
    root = ExplorerDOM(blind.text).root
    assert (
        root.by_id("team-hidden") is not None and root.by_id("protocol-blind-state").text() == "on"
    )
    assert not root.by_class("conflict-item") and not root.by_class("agreement-line")
    assert values(root, "progress-reviewer") == ["r1", "r2"]
    protocol = form_fields(root, "protocol-form")
    assert protocol["expected_protocol_revision"] == "0"
    changed = team.client.post(
        PROTOCOL,
        data={
            **protocol,
            "blind": "off",
            "exclusion_reasons_must_match": "no",
            "changed_by": "lead",
        },
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert changed.status_code == 303
    opened_page = page(team, changed.headers["location"])
    assert opened_page.by_id("protocol-saved").text() == "Protocol revision 1 confirmed."
    lines = values(opened_page, "agreement-line")
    assert lines[0].startswith("n 6, p_o 0.500, p_e 0.444, kappa 0.100 (95% CI")
    assert len(opened_page.by_class("conflict-item")) == 3
    assert opened_page.by_id("unresolved-conflicts").text() == "3"
    for identifier in ("agreement-json", "agreement-markdown", "agreement-csv", "final-csv"):
        assert team.client.get(href(opened_page, f"download-{identifier}")).status_code == 200
    resolution = form_fields(opened_page, "resolution-form-1")
    resolved = team.client.post(
        TEAM,
        data={
            "selection": resolution["selection"],
            "expected_resolution_revision": resolution["expected_resolution_revision"],
            "resolver": "lead",
            "decision": "exclude",
            "rationale": "Team decision after discussion",
        },
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert resolved.status_code == 303
    after = page(team, resolved.headers["location"])
    assert after.by_id("resolution-saved") is not None
    assert after.by_id("unresolved-conflicts").text() == "2"
    assert after.by_id("resolved-conflicts").text() == "1"
    assert values(after, "resolution-rationale") == ["Team decision after discussion"]
    stale = team.client.post(
        TEAM,
        data={
            "selection": resolution["selection"],
            "expected_resolution_revision": "0",
            "resolver": "lead",
            "decision": "include",
            "rationale": "Stale form",
        },
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert stale.status_code == 409 and "Stale form" not in stale.text


@pytest.mark.parametrize(
    ("method", "path", "allow"),
    [("PUT", REVIEWER, "GET, POST"), ("DELETE", TEAM, "GET, POST"), ("GET", PROTOCOL, "POST")],
)
def test_wrong_methods_are_rejected_with_allow(
    team: Team, method: str, path: str, allow: str
) -> None:
    response = team.client.request(method, path)
    assert response.status_code == 405 and response.headers["allow"] == allow


@pytest.mark.parametrize(
    "params",
    [
        {"reviewer": "default"},
        {"reviewer": "Reviewer One"},
        {"document_id": IDS[0]},
        {"reviewer": "r2", "document_id": IDS[0], "order": "priority"},
        {"reviewer": "r2", "blind": "false"},
        {"reviewer": "r2", "unknown": "PRIVATE_FIELD"},
    ],
)
def test_invalid_reviewer_state_fails_without_echo(team: Team, params: dict[str, str]) -> None:
    response = team.client.get(REVIEWER, params=opened(team, **params))
    assert response.status_code == 422
    assert "PRIVATE_FIELD" not in response.text and "Reviewer One" not in response.text
    assert not ExplorerDOM(response.text).root.by_tag("form")


def test_mounted_pages_keep_root_path(team: Team) -> None:
    with TestClient(team.app, root_path="/library") as client:
        response = client.get(f"/library{REVIEWER}", params=opened(team, reviewer="r2"))
        assert response.status_code == 200
        root = ExplorerDOM(response.text).root
        for anchor in root.by_tag("a"):
            parts = urlsplit(anchor.attrs["href"])
            assert not parts.scheme and not parts.netloc
            assert not parts.path or parts.path.startswith("/library/")
        team_page = ExplorerDOM(client.get(f"/library{TEAM}", params=opened(team)).text).root
        assert team_page.by_id("protocol-form").attrs["action"] == f"/library{PROTOCOL}"
