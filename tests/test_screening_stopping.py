"""Advisory screening stopping rules: exact math, recorded order, heuristics and buscar."""

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from scripts.demo_evidence_export import offline_settings

from api.application import create_app
from api.dependencies import AppContainer
from retrieval.models import Document
from storage import paper_screening
from storage import screening_stopping as stopping
from storage.paper_collections import CollectionEdit, CollectionError, PaperCollection
from storage.paper_screening import ScreeningSubmission
from storage.screening_stopping import (
    UrnObservation,
    assess_screening_stopping,
    hypergeometric_cdf,
    hypergeometric_pmf,
    min_relevant_for_missed_target,
    recall_estimate,
    recall_lower_bound,
    recall_p_value,
    sample_order,
    screens_until_met,
    urn_digest,
)

BASE = datetime(2026, 11, 24, 9, 0, tzinfo=UTC)
COLLECTION = "col_" + "0" * 32


def test_hypergeometric_matches_hand_computed_values() -> None:
    # C(3,1) C(7,3) / C(10,4) = 3 * 35 / 210.
    assert hypergeometric_pmf(1, 10, 3, 4) == Fraction(1, 2)
    # P(X <= 1) = (C(7,4) + 3 * C(7,3)) / C(10,4) = (35 + 105) / 210.
    assert hypergeometric_cdf(1, 10, 3, 4) == Fraction(2, 3)
    # No relevant record in 20 draws from 50 with 2 relevant: (30 * 29) / (50 * 49).
    assert hypergeometric_cdf(0, 50, 2, 20) == Fraction(87, 245)
    assert sum(hypergeometric_pmf(k, 12, 5, 7) for k in range(8)) == 1
    # With 8 relevant among 10, five draws contain at least three relevant records.
    assert hypergeometric_pmf(2, 10, 8, 5) == 0 and hypergeometric_cdf(2, 10, 8, 5) == 0
    assert hypergeometric_cdf(5, 10, 8, 5) == 1 and hypergeometric_cdf(0, 10, 3, 0) == 1


@pytest.mark.parametrize(
    "arguments",
    [(0, 5, 6, 1), (0, 5, 1, 6), (0, -1, 0, 0), (0, 5, 1.0, 1), (0, 5, True, 1), (0.5, 5, 1, 1)],
)
def test_hypergeometric_rejects_invalid_arguments(arguments: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        hypergeometric_cdf(*arguments)
    with pytest.raises(ValueError):
        hypergeometric_pmf(*arguments)


def test_buscar_k_hat_uses_the_exact_decimal_target() -> None:
    # floor(19 / 0.95 - 19) + 1 = floor(20 - 19) + 1.
    assert min_relevant_for_missed_target(19, 19, Fraction(19, 20)) == 2
    assert min_relevant_for_missed_target(4, 4, Fraction(19, 20)) == 1
    assert min_relevant_for_missed_target(0, 0, Fraction(19, 20)) == 1
    # Target 1 asks whether any relevant record remains beyond those already found.
    assert min_relevant_for_missed_target(5, 3, Fraction(1)) == 3
    # 7 / 0.035 is exactly 200; binary floats give 199.99999999999997 and one fewer.
    assert min_relevant_for_missed_target(7, 7, Fraction("0.035")) == 194
    for target, before in ((Fraction(0), 0), (Fraction(3, 2), 0), (Fraction(1, 2), 6)):
        with pytest.raises(ValueError):
            min_relevant_for_missed_target(5, before, target)


def test_recall_p_value_matches_hand_computed_buscar_examples() -> None:
    target = Fraction(19, 20)
    after_twenty = UrnObservation(
        urn_size=50, sample_size=20, sample_relevant=0, relevant_before=19
    )
    assert recall_p_value(after_twenty, target) == (2, Fraction(87, 245))
    # K_hat = floor(19 / 0.95 - 18) + 1 = 3; P(X <= 1 | 40, 3, 20) = 3/26 + 5/13.
    one_relevant = UrnObservation(
        urn_size=40, sample_size=20, sample_relevant=1, relevant_before=18
    )
    assert recall_p_value(one_relevant, target) == (3, Fraction(1, 2))
    exhausted = UrnObservation(urn_size=5, sample_size=5, sample_relevant=0, relevant_before=2)
    # K_hat = floor(2 / 0.95 - 2) + 1 = 1, but no urn member can still be relevant.
    assert recall_p_value(exhausted, target) == (1, Fraction(0))
    # One urn member was decided relevant outside the sample: K_hat 2 exceeds capacity 1.
    outside = UrnObservation(
        urn_size=30,
        sample_size=29,
        sample_relevant=0,
        relevant_before=0,
        decided_outside=1,
        relevant_outside=1,
    )
    assert outside.capacity == 1 and recall_p_value(outside, target) == (2, Fraction(0))
    assert recall_lower_bound(outside, Fraction(1, 20)) == 1
    with pytest.raises(ValueError):
        UrnObservation(
            urn_size=3, sample_size=3, sample_relevant=0, relevant_before=0, decided_outside=1
        )


def test_recall_lower_bound_and_estimate_match_hand_computation() -> None:
    alpha = Fraction(1, 20)
    observation = UrnObservation(urn_size=50, sample_size=20, sample_relevant=0, relevant_before=19)
    # P(X = 0) for K = 5 is 0.067 >= 0.05 and for K = 6 it is 0.037: at most 5 remain.
    assert hypergeometric_cdf(0, 50, 5, 20) >= alpha > hypergeometric_cdf(0, 50, 6, 20)
    assert recall_lower_bound(observation, alpha) == Fraction(19, 24)
    assert recall_estimate(observation) == 1
    sampled = UrnObservation(urn_size=50, sample_size=20, sample_relevant=2, relevant_before=18)
    # 2/20 of the 30 unscreened urn members: 3 expected relevant remain.
    assert recall_estimate(sampled) == Fraction(20, 23)
    nothing = UrnObservation(urn_size=10, sample_size=4, sample_relevant=0, relevant_before=0)
    assert recall_estimate(nothing) is None and recall_lower_bound(nothing, alpha) is None
    unsampled = UrnObservation(urn_size=10, sample_size=0, sample_relevant=0, relevant_before=3)
    assert recall_estimate(unsampled) is None
    assert recall_p_value(unsampled, Fraction(19, 20))[1] == 1
    with pytest.raises(ValueError):
        recall_lower_bound(observation, Fraction(1))


def test_rejecting_the_target_is_exactly_the_lower_bound_reaching_it() -> None:
    alpha = Fraction(1, 20)
    checked = 0
    for urn in range(1, 9):
        for sample in range(urn + 1):
            for relevant in range(sample + 1):
                for before in range(4):
                    for outside, outside_relevant in ((0, 0), (1, 0), (1, 1)):
                        if sample + outside > urn:
                            continue
                        observation = UrnObservation(
                            urn_size=urn,
                            sample_size=sample,
                            sample_relevant=relevant,
                            relevant_before=before,
                            decided_outside=outside,
                            relevant_outside=outside_relevant,
                        )
                        bound = recall_lower_bound(observation, alpha)
                        for target in (
                            Fraction(1, 2),
                            Fraction(4, 5),
                            Fraction(19, 20),
                            Fraction(1),
                        ):
                            rejected = recall_p_value(observation, target)[1] < alpha
                            if bound is not None:
                                assert rejected == (bound >= target), observation
                                checked += 1
    assert checked > 3000


def test_projection_counts_only_pending_sample_records() -> None:
    target, alpha = Fraction(19, 20), Fraction(1, 20)
    observation = UrnObservation(urn_size=10, sample_size=6, sample_relevant=0, relevant_before=3)
    # p(n) = (10 - n) / 10 with K_hat = 1: only n = 10 is below 0.05.
    assert screens_until_met(observation, ["unscreened"] * 4, target, alpha) == 4
    assert screens_until_met(observation, ["unscreened"] * 4, target, Fraction(1, 5)) == 3
    with_outside = UrnObservation(
        urn_size=9, sample_size=6, sample_relevant=0, relevant_before=3, decided_outside=1
    )
    # p: 3/9, 2/9, then 1/9 when the outside exclude joins, then 0: two screened.
    assert screens_until_met(with_outside, ["stale", "exclude", "unscreened"], target, alpha) == 2
    assert screens_until_met(observation, ["unscreened", "unsure"], target, alpha) is None
    done = UrnObservation(urn_size=4, sample_size=4, sample_relevant=0, relevant_before=3)
    assert screens_until_met(done, [], target, alpha) == 0


def test_sample_order_is_the_documented_sha256_order() -> None:
    members = [f"paper-{index:02d}" for index in range(12)]

    def documented(seed: int) -> list[str]:
        def key(identifier: str) -> str:
            material = json.dumps(
                ["scholar-rag-stopping-sample-v1", seed, COLLECTION, 3, identifier],
                ensure_ascii=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(material.encode("ascii")).hexdigest()

        return sorted(members, key=key)

    order = sample_order(members, seed=20261124, collection_id=COLLECTION, collection_revision=3)
    assert list(order) == documented(20261124)
    assert order == sample_order(
        reversed(members), seed=20261124, collection_id=COLLECTION, collection_revision=3
    )
    assert sorted(order) == members and order != tuple(members)
    assert order != sample_order(members, seed=1, collection_id=COLLECTION, collection_revision=3)
    assert urn_digest(members, collection_id=COLLECTION, collection_revision=3) == urn_digest(
        reversed(members), collection_id=COLLECTION, collection_revision=3
    )
    assert urn_digest(members, collection_id=COLLECTION, collection_revision=3) != urn_digest(
        members[1:], collection_id=COLLECTION, collection_revision=3
    )


class Clock:
    """Replaces paper_screening.datetime so every decision has a chosen recorded time."""

    def __init__(self) -> None:
        self.current = BASE

    def now(self, tz: object = None) -> datetime:
        return self.current


@dataclass
class Review:
    container: AppContainer
    database: Path
    collection: PaperCollection
    clock: Clock
    revisions: dict[str, int] = field(default_factory=dict)

    @property
    def members(self) -> tuple[str, ...]:
        return self.collection.document_ids

    def decide(self, identifier: str, decision: str, minute: float) -> None:
        self.clock.current = BASE + timedelta(minutes=minute)
        self.container.paper_screening.submit(
            self.collection.collection_id,
            identifier,
            ScreeningSubmission.model_validate(
                {
                    "collection_revision": self.collection.revision,
                    "expected_decision_revision": self.revisions.get(identifier, 0),
                    "decision": decision,
                    "reason": "Synthetic human opinion for a stopping-rule test.",
                }
            ),
        )
        self.revisions[identifier] = self.revisions.get(identifier, 0) + 1

    def assess(self, **parameters: object) -> stopping.ScreeningStoppingAssessment:
        parameters.setdefault("collection_revision", self.collection.revision)
        return assess_screening_stopping(
            self.container.paper_screening, self.collection.collection_id, **parameters
        )

    def fails(self, status: int, code: str, **parameters: object) -> None:
        with pytest.raises(CollectionError) as caught:
            self.assess(**parameters)
        assert (caught.value.status_code, caught.value.code) == (status, code)


def _review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int) -> Review:
    clock = Clock()
    monkeypatch.setattr(paper_screening, "datetime", clock)
    database = tmp_path / "stopping.sqlite3"
    container: AppContainer = create_app(offline_settings(database)).state.container
    identifiers = [f"p{index:02d}" for index in range(size)]
    container.document_store.add_documents(
        [
            Document(document_id=identifier, title=f"Paper {identifier}", text="x", source="s")
            for identifier in identifiers
        ],
        [],
    )
    collection = container.paper_collections.create(name="Stopping", document_ids=identifiers)
    return Review(container, database, collection, clock)


@pytest.fixture
def review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Review]:
    yield _review(tmp_path, monkeypatch, 8)


def test_nothing_screened_reports_every_rule_without_inventing_a_signal(review: Review) -> None:
    result = review.assess()
    assert result.advisory_only and not result.all_screened
    assert result.timeline.items == () and result.timeline.pending == 8
    rule = result.consecutive_irrelevant
    assert (rule.status, rule.current_run, rule.threshold, rule.last_relevant_at) == (
        "not_met",
        0,
        1,
        None,
    )
    assert result.parameters.irrelevant_run_basis == "percent_of_collection"
    assert result.parameters.irrelevant_run_percent == 10
    assert result.safe_checkpoint.status == "not_met"
    test = result.recall_test
    assert (test.status, test.sample_after, test.urn_size, test.p_value) == (
        "not_started",
        0,
        8,
        None,
    )
    assert test.urn_sha256 == urn_digest(
        review.members, collection_id=review.collection.collection_id, collection_revision=1
    )
    assert any("wall-clock" in text for text in result.assumptions)
    assert any("Advisory only" in text for text in result.limitations)


def test_recorded_order_and_run_use_the_conservative_placement(review: Review) -> None:
    for identifier, decision, minute in (
        ("p00", "include", 1),
        ("p01", "exclude", 2),
        ("p02", "unsure", 3),
        ("p03", "exclude", 4),
        ("p04", "exclude", 5),
        ("p05", "exclude", 6),
        ("p02", "exclude", 10),
        ("p04", "include", 11),
        ("p06", "exclude", 12),
        ("p07", "exclude", 13),
    ):
        review.decide(identifier, decision, minute)
    result = review.assess(irrelevant_run=3)
    timeline = result.timeline
    assert [item.document_id for item in timeline.items] == list(review.members)
    assert [item.includes_so_far for item in timeline.items] == [1, 1, 1, 1, 2, 2, 2, 2]
    resolved = timeline.items[2]
    assert (resolved.decision, resolved.first_decided_at, resolved.decided_at) == (
        "exclude",
        BASE + timedelta(minutes=3),
        BASE + timedelta(minutes=10),
    )
    assert timeline.order_reliable and timeline.tied_positions == 0
    rule = result.consecutive_irrelevant
    # The latest relevant decision is p04's include at minute 11, so only p06 and p07 count.
    assert rule.last_relevant_at == BASE + timedelta(minutes=11)
    assert rule.run_document_ids == ("p06", "p07")
    assert (rule.status, rule.current_run, rule.threshold_percent_of_collection) == (
        "not_met",
        2,
        37.5,
    )
    assert review.assess(irrelevant_run_percent=25).consecutive_irrelevant.status == "met"
    assert review.assess().consecutive_irrelevant.threshold == 1
    assert result.parameters.irrelevant_run_basis == "records"


def test_equal_times_never_lengthen_the_run(review: Review) -> None:
    review.decide("p00", "exclude", 5)
    review.decide("p01", "include", 5)
    review.decide("p02", "exclude", 6)
    result = review.assess()
    assert [item.tied_with_previous for item in result.timeline.items] == [False, True, False]
    assert result.timeline.tied_positions == 1
    assert result.consecutive_irrelevant.run_document_ids == ("p02",)


def test_safe_checkpoint_needs_every_condition(review: Review) -> None:
    for index, decision in enumerate(("include", "include", "exclude", "exclude")):
        review.decide(f"p0{index}", decision, index)
    result = review.assess()
    names = [criterion.name for criterion in result.safe_checkpoint.criteria]
    assert names == [
        "key_papers_included",
        "screened_twice_estimated_relevant",
        "minimum_screened_percent",
        "consecutive_irrelevant",
    ]
    assert [criterion.status for criterion in result.safe_checkpoint.criteria] == [
        "not_evaluable",
        "not_evaluable",
        "met",
        "met",
    ]
    assert result.safe_checkpoint.status == "incomplete"
    complete = review.assess(key_document_ids=["p00", "p01"], estimated_relevant=2)
    assert complete.safe_checkpoint.status == "met"
    missing = review.assess(key_document_ids=["p01", "p07"], estimated_relevant=3)
    assert missing.safe_checkpoint.status == "not_met"
    assert missing.safe_checkpoint.key_documents_not_included == ("p07",)
    assert [criterion.status for criterion in missing.safe_checkpoint.criteria[:2]] == [
        "not_met",
        "not_met",
    ]
    assert review.assess(min_screened_percent=60).safe_checkpoint.criteria[2].status == "not_met"


def test_seeded_sample_flow_reaches_the_target_exactly_as_computed_by_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review(tmp_path, monkeypatch, 20)
    prioritized = ["include"] * 4 + ["exclude"] * 6
    for index, decision in enumerate(prioritized):
        review.decide(f"p{index:02d}", decision, index)
    start = review.assess().recall_test
    assert (start.status, start.sample_after, start.urn_size) == ("not_started", 10, 10)
    parameters = {"sample_seed": 7, "sample_after": 10, "sample_urn_sha256": start.urn_sha256}
    drawn = review.assess(**parameters).recall_test
    urn = [f"p{index:02d}" for index in range(10, 20)]
    order = sample_order(
        urn, seed=7, collection_id=review.collection.collection_id, collection_revision=1
    )
    assert [item.document_id for item in drawn.sample_order] == list(order)
    assert drawn.next_document_ids == order and drawn.urn_sha256 == start.urn_sha256
    # K_hat = floor(4 / 0.95 - 4) + 1 = 1, so p(n) = P(no relevant in n of 10) = (10 - n) / 10.
    assert (drawn.status, drawn.h0_min_relevant_in_urn, drawn.p_value_exact) == (
        "not_met",
        1,
        "1/1",
    )
    assert drawn.additional_irrelevant_needed == 10
    for screened, identifier in enumerate(order[:9], 1):
        review.decide(identifier, "exclude", 30 + screened)
        test = review.assess(**parameters).recall_test
        assert (test.sample_size, test.sample_relevant, test.relevant_seen) == (screened, 0, 4)
        assert test.p_value_exact == str(Fraction(10 - screened, 10))
        assert test.additional_irrelevant_needed == 10 - screened
        assert test.recall_estimate == 1.0
        assert test.next_document_ids == order[screened:]
    assert test.status == "not_met" and test.p_value == pytest.approx(0.1)
    looser = review.assess(confidence=0.8, **parameters).recall_test
    assert (looser.status, looser.alpha) == ("met", pytest.approx(0.2))
    review.decide(order[9], "exclude", 40)
    final = review.assess(**parameters)
    assert final.recall_test.status == "met" and final.recall_test.p_value == 0
    assert final.recall_test.recall_lower_bound == 1.0 and final.all_screened
    assert "reject H0" in final.recall_test.explanation


def test_unsure_pauses_the_sample_and_outside_decisions_count_as_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review(tmp_path, monkeypatch, 12)
    for index, decision in enumerate(["include", "include", "exclude", "unsure"]):
        review.decide(f"p{index:02d}", decision, index)
    urn = [f"p{index:02d}" for index in range(4, 12)]
    order = sample_order(
        urn, seed=11, collection_id=review.collection.collection_id, collection_revision=1
    )
    review.decide(order[0], "exclude", 20)
    review.decide(order[1], "unsure", 21)
    review.decide(order[5], "include", 22)
    review.decide(order[6], "exclude", 23)
    test = review.assess(sample_seed=11, sample_after=4).recall_test
    assert test.blocked_by_unsure == order[1]
    assert (test.sample_size, test.sample_relevant, test.relevant_before_sample) == (1, 0, 2)
    assert (test.decided_outside_sample, test.relevant_outside_sample, test.relevant_seen) == (
        2,
        1,
        3,
    )
    assert test.next_document_ids == (order[1], *order[2:5], order[7])
    assert test.additional_irrelevant_needed is None
    # K_hat = floor(3 / 0.95 - 2) + 1 = 2; P(X <= 0 | 8, 2, 1) = 6/8.
    assert (test.h0_min_relevant_in_urn, test.p_value_exact) == (2, "3/4")
    assert "is unsure" in test.explanation
    review.decide(order[1], "exclude", 24)
    resumed = review.assess(sample_seed=11, sample_after=4).recall_test
    assert resumed.blocked_by_unsure is None and resumed.sample_size == 2
    # The unsure prioritized-phase member ends no relevant count, but it ends the run.
    assert review.assess().consecutive_irrelevant.last_relevant_at is not None


def test_zero_relevant_and_all_screened_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review(tmp_path, monkeypatch, 4)
    for index in range(3):
        review.decide(f"p{index:02d}", "exclude", index)
    nothing = review.assess(sample_seed=3, sample_after=0)
    assert nothing.consecutive_irrelevant.current_run == 3
    assert "no include or unsure decision yet" in nothing.consecutive_irrelevant.explanation
    test = nothing.recall_test
    # Nothing relevant found: K_hat = 1 and recall stays undefined.
    assert test.h0_min_relevant_in_urn == 1 and test.relevant_seen == 0
    assert test.recall_estimate is None and test.recall_lower_bound is None
    review.decide("p03", "exclude", 9)
    finished = review.assess()
    assert finished.all_screened and finished.recall_test.urn_size == 0
    empty_urn = review.assess(sample_seed=3, sample_after=4).recall_test
    assert (empty_urn.status, empty_urn.p_value, empty_urn.sample_order) == ("met", 0.0, ())
    everything = review.assess(sample_seed=3, sample_after=0).recall_test
    assert everything.status == "met" and everything.sample_size == 4


@pytest.mark.parametrize(
    "parameters",
    [
        {"irrelevant_run": 2, "irrelevant_run_percent": 5},
        {"sample_seed": 1},
        {"sample_after": 0},
        {"sample_urn_sha256": "0" * 64},
        {"sample_seed": 1, "sample_after": 0, "sample_urn_sha256": "A" * 64},
        {"key_document_ids": ["p00", "p00"]},
        {"key_document_ids": [" p00"]},
        {"target_recall": 0},
        {"target_recall": 1.5},
        {"target_recall": 0.9500001},
        {"confidence": 1},
        {"confidence": float("nan")},
        {"irrelevant_run": 0},
        {"irrelevant_run": 101},
        {"irrelevant_run_percent": 101},
        {"min_screened_percent": 0},
        {"estimated_relevant": -1},
        {"sample_seed": -1, "sample_after": 0},
        {"sample_seed": 2**63, "sample_after": 0},
        {"irrelevant_run": True},
    ],
)
def test_invalid_parameters_fail_before_any_read(
    review: Review, parameters: dict[str, Any]
) -> None:
    review.fails(422, "invalid_screening_stopping_request", **parameters)


def test_parameters_are_checked_against_the_snapshot(review: Review) -> None:
    review.decide("p00", "include", 1)
    review.fails(422, "stopping_key_document_not_member", key_document_ids=["missing"])
    review.fails(422, "stopping_parameter_out_of_range", estimated_relevant=9)
    review.fails(422, "stopping_parameter_out_of_range", irrelevant_run=9)
    review.fails(422, "stopping_parameter_out_of_range", sample_seed=1, sample_after=2)
    review.fails(409, "collection_revision_conflict", collection_revision=2)
    with pytest.raises(CollectionError) as caught:
        assess_screening_stopping(
            review.container.paper_screening, COLLECTION, collection_revision=1
        )
    assert caught.value.code == "collection_not_found"


def test_ambiguous_or_changed_urns_are_refused(review: Review) -> None:
    review.decide("p00", "include", 1)
    review.decide("p01", "exclude", 2)
    review.decide("p02", "exclude", 2)
    review.fails(409, "stopping_sample_boundary_tied", sample_seed=1, sample_after=2)
    start = review.assess().recall_test
    review.decide("p03", "exclude", 3)
    review.fails(
        409,
        "stopping_sample_urn_changed",
        sample_seed=1,
        sample_after=4,
        sample_urn_sha256=start.urn_sha256,
    )
    assert review.assess(
        sample_seed=1, sample_after=3, sample_urn_sha256=start.urn_sha256
    ).recall_test.status in ("met", "not_met")


def test_a_clock_moving_backwards_makes_order_rules_not_evaluable(review: Review) -> None:
    review.decide("p00", "include", 1)
    review.decide("p01", "exclude", 10)
    review.decide("p01", "exclude", 5)
    review.decide("p02", "exclude", 11)
    result = review.assess(sample_seed=4, sample_after=2)
    assert not result.timeline.order_reliable
    assert result.timeline.order_warnings == (
        'Member "p01": decision revision 2 has an earlier recorded time than revision 1.',
    )
    assert result.consecutive_irrelevant.status == "not_evaluable"
    assert result.safe_checkpoint.criteria[3].status == "not_evaluable"
    assert result.recall_test.status == "not_evaluable" and result.recall_test.p_value is None
    assert "sample_urn_sha256" in result.recall_test.explanation
    urn = urn_digest(
        [f"p{index:02d}" for index in range(2, 8)],
        collection_id=review.collection.collection_id,
        collection_revision=1,
    )
    verified = review.assess(sample_seed=4, sample_after=2, sample_urn_sha256=urn).recall_test
    assert verified.status == "not_met" and verified.p_value is not None


def test_a_decision_from_an_older_revision_after_a_current_one_is_flagged(
    review: Review,
) -> None:
    review.decide("p00", "include", 10)
    review.container.document_store.add_documents(
        [Document(document_id="p08", title="Added", text="x", source="s")], []
    )
    review.collection = review.container.paper_collections.edit(
        review.collection.collection_id,
        changes=CollectionEdit(name="Stopping", expected_revision=1, add_document_ids=("p08",)),
    )
    review.decide("p01", "exclude", 5)
    review.decide("p00", "include", 20)
    result = review.assess()
    assert result.collection_revision == 2 and result.counts.stale == 0
    assert [item.document_id for item in result.timeline.items] == ["p01", "p00"]
    assert result.timeline.items[1].first_decision_retained
    assert result.timeline.order_warnings == (
        'Member "p00" has a decision from an earlier collection revision recorded after '
        'member "p01" was decided at revision 2.',
    )


def test_stale_members_are_pending_and_unretained_history_never_lengthens_a_run(
    review: Review,
) -> None:
    review.decide("p00", "include", 1)
    review.decide("p01", "exclude", 2)
    legacy = paper_screening.ScreeningReview(
        schema_version="1.0",
        collection_id=review.collection.collection_id,
        document_id="p02",
        collection_revision=1,
        revision=2,
        decision="exclude",
        reason="Saved before decision history existed.",
        updated_at=BASE + timedelta(minutes=30),
    )
    with closing(sqlite3.connect(review.database)) as connection, connection:
        connection.execute(
            "INSERT INTO paper_screening_decisions VALUES (?, ?, ?)",
            (review.collection.collection_id, "p02", legacy.model_dump_json()),
        )
    result = review.assess()
    item = result.timeline.items[-1]
    assert (item.document_id, item.first_decision_retained) == ("p02", False)
    assert result.consecutive_irrelevant.run_document_ids == ("p01",)


def test_assessment_is_read_only(review: Review, monkeypatch: pytest.MonkeyPatch) -> None:
    review.decide("p00", "include", 1)
    review.decide("p01", "exclude", 2)
    before = review.database.read_bytes()
    monkeypatch.setattr(
        review.container.paper_screening,
        "submit",
        lambda *args, **kwargs: pytest.fail("A stopping assessment must not record decisions."),
    )
    first = review.assess(sample_seed=9, sample_after=2)
    assert first == review.assess(sample_seed=9, sample_after=2)
    assert review.database.read_bytes() == before
    assert review.container.event_log.list_events() == []


def test_bounds_and_invalid_saved_history_fail_without_partial_results(
    review: Review, monkeypatch: pytest.MonkeyPatch
) -> None:
    review.decide("p00", "include", 1)
    review.decide("p00", "exclude", 2)
    monkeypatch.setattr(stopping, "MAX_HISTORY_ROWS", 1)
    review.fails(413, "screening_stopping_history_too_large")
    monkeypatch.setattr(stopping, "MAX_HISTORY_ROWS", 10_000)
    monkeypatch.setattr(stopping, "MAX_STOPPING_RESPONSE_BYTES", 100)
    review.fails(413, "screening_stopping_too_large")
    monkeypatch.setattr(stopping, "MAX_STOPPING_RESPONSE_BYTES", 524_288)
    with closing(sqlite3.connect(review.database)) as connection, connection:
        connection.execute("DELETE FROM paper_screening_history WHERE revision = 1")
    assert review.assess().timeline.items[0].first_decision_retained is False
    with closing(sqlite3.connect(review.database)) as connection, connection:
        connection.execute("UPDATE paper_screening_history SET record = '{bad'")
    review.fails(409, "invalid_screening_history_record")


def test_history_is_paged_and_must_end_at_the_current_decision(
    review: Review, monkeypatch: pytest.MonkeyPatch
) -> None:
    for minute, decision in enumerate(("unsure", "exclude", "include", "exclude"), 1):
        review.decide("p00", decision, minute)
    monkeypatch.setattr(stopping, "_HISTORY_PAGE", 1)
    item = review.assess().timeline.items[0]
    assert (item.first_decided_at, item.decided_at) == (
        BASE + timedelta(minutes=1),
        BASE + timedelta(minutes=4),
    )
    assert item.first_decision_retained
    with closing(sqlite3.connect(review.database)) as connection, connection:
        connection.execute(
            "UPDATE paper_screening_decisions SET record = replace(record, ?, ?)",
            ("Synthetic human opinion", "Edited outside the app"),
        )
    review.fails(409, "invalid_screening_history_record")


def test_the_pure_builder_refuses_an_inconsistent_snapshot(review: Review) -> None:
    review.decide("p00", "include", 1)
    with review.container.paper_collections.resolved_collection(
        review.collection.collection_id
    ) as (connection, collection):
        queue = review.container.paper_screening._snapshot(connection, collection)
    options = stopping.StoppingOptions(
        collection_id=collection.collection_id, collection_revision=1
    )
    with pytest.raises(CollectionError) as caught:
        stopping.build_assessment(collection, queue, [], [], options)
    assert caught.value.code == "invalid_screening_stopping"
