"""Pure TF-IDF + Naive Bayes ordering of the human screening queue, never a decision."""

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.screening_priority_fixtures import collection_order, synthetic_papers

from retrieval.models import Document
from storage import screening_priority
from storage.document_store import SQLiteDocumentStore
from storage.paper_collections import CollectionError, PaperCollection
from storage.paper_screening import ScreeningSubmission, SQLitePaperScreening
from storage.screening_priority import (
    PriorityMember,
    ScreeningPriority,
    SimulatedPaper,
    build_priority,
    explain,
    feature_text,
    prioritize_screening,
    rank_members,
    relevance_probability,
    simulate_screening,
    tfidf_vectors,
    tokenize,
    train_naive_bayes,
)


def member(identifier: str, status: str, title: str, text: str = "") -> PriorityMember:
    return PriorityMember(
        document_id=identifier,
        status=status,  # type: ignore[arg-type]
        title=title,
        title_truncated=False,
        text=text,
    )


def test_tokenizer_casefolds_and_drops_stop_words_digits_short_and_long_tokens() -> None:
    text = "The CAFE\u0301 study_of RAG-based 2024 Q&A x " + "y" * 65 + " covid19 Stra\u00dfe"
    assert tokenize(text) == ["café", "study", "rag", "based", "covid19", "strasse"]
    assert tokenize("of the and is 12 345") == []
    assert feature_text("t" * 400, "x" * 5000) == "t" * 300 + "\n" + "x" * 4000


def test_tfidf_and_naive_bayes_match_hand_computed_formulas() -> None:
    documents = [["alpha", "beta"], ["gamma", "beta"], ["alpha", "gamma", "alpha"]]
    vectors = tfidf_vectors(documents)
    idf = math.log(4 / 3) + 1
    assert vectors[0] == pytest.approx({"alpha": 2**-0.5, "beta": 2**-0.5})
    alpha_weight, gamma_weight = (1 + math.log(2)) * idf, idf
    norm = math.hypot(alpha_weight, gamma_weight)
    assert vectors[2] == pytest.approx({"alpha": alpha_weight / norm, "gamma": gamma_weight / norm})
    weights = train_naive_bayes([vectors[0]], [vectors[1]])
    # Each class has one paper: theta = (mass + 1) / (sqrt(2) + 3) for alpha, beta, gamma.
    denominator = math.sqrt(2) + 3
    expected = math.log((2**-0.5 + 1) / denominator) - math.log(1 / denominator)
    assert weights == pytest.approx({"alpha": expected, "beta": 0.0, "gamma": -expected})
    log_odds, contributions = explain(vectors[2], weights)
    assert log_odds == pytest.approx(expected * (alpha_weight - gamma_weight) / norm)
    assert [term for term, _ in contributions] == ["alpha", "gamma"]
    assert relevance_probability(log_odds) == pytest.approx(1 / (1 + math.exp(-log_odds)))
    assert relevance_probability(-800.0) == 0.0 and relevance_probability(800.0) == 1.0


def test_class_balanced_masses_use_means_and_reject_missing_classes() -> None:
    one, two = {"a": 1.0}, {"a": 0.5, "b": 0.5}
    weights = train_naive_bayes([one, two], [{"c": 1.0}], alpha=0.5)
    mean = {"a": 0.75, "b": 0.25, "c": 0.0}
    relevant_total = sum(mean.values()) + 1.5
    irrelevant_total = 1.0 + 1.5
    assert weights["b"] == pytest.approx(
        math.log((0.25 + 0.5) / relevant_total) - math.log(0.5 / irrelevant_total)
    )
    for invalid in ({"relevant": [], "irrelevant": [one]}, {"relevant": [one], "irrelevant": []}):
        with pytest.raises(ValueError):
            train_naive_bayes(**invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        train_naive_bayes([one], [two], alpha=0.0)


def test_rank_orders_pending_members_and_explains_with_discriminative_terms() -> None:
    members = [
        member("unread-low", "unscreened", "Battery cathode coatings"),
        member("seed-include", "include", "Citation grounding for evidence answers"),
        member("seed-exclude", "exclude", "Battery cathode degradation"),
        member("unread-high", "unscreened", "Evidence citation grounding in answers"),
        member("unread-mixed", "unscreened", "Citation evidence for battery cathodes"),
    ]
    ranking = rank_members(members)
    assert ranking.model_status == "ready"
    assert [item.document_id for item in ranking.items] == [
        "unread-high",
        "unread-mixed",
        "unread-low",
    ]
    assert [item.rank for item in ranking.items] == [1, 2, 3]
    top = ranking.items[0]
    assert top.score is not None and top.log_odds is not None and top.score > 0.5
    terms = [term.term for term in top.top_terms]
    assert {"citation", "grounding", "evidence"} <= set(terms)
    assert "battery" not in terms and "cathode" not in terms
    assert all(term.contribution > 0 for term in top.top_terms)
    contributions = [term.contribution for term in top.top_terms]
    assert contributions == sorted(contributions, reverse=True)
    low = ranking.items[-1]
    assert low.score is not None and low.score < 0.5 and low.top_terms == ()
    assert ranking.feature_terms == len({"citation", "grounding", "evidence", "answers"}) + 3


def test_equal_scores_break_ties_by_document_id_deterministically() -> None:
    members = [
        member("z-label", "include", "Grounded citation"),
        member("y-label", "exclude", "Battery cathode"),
        member("b-duplicate", "unscreened", "Grounded citation"),
        member("a-duplicate", "unscreened", "Grounded citation"),
        member("c-neutral", "unscreened", "Unrelated words entirely"),
    ]
    first = rank_members(members)
    second = rank_members(list(reversed(members)))
    assert [item.document_id for item in first.items] == [
        "a-duplicate",
        "b-duplicate",
        "c-neutral",
    ]
    assert first.items[0].log_odds == first.items[1].log_odds
    assert [item.document_id for item in second.items] == [item.document_id for item in first.items]
    assert first.items[2].log_odds == 0.0 and first.items[2].score == 0.5


def test_unsure_and_stale_decisions_are_never_training_labels() -> None:
    base = [
        member("include", "include", "Grounded citation evidence"),
        member("exclude", "exclude", "Battery cathode"),
        member("pending", "unscreened", "Citation evidence"),
        member("stale", "stale", "Battery stale unique opinion"),
        member("unsure", "unsure", "Grounded unsure distinctive wording"),
    ]
    relabeled = [
        member(m.document_id, "unscreened", m.title) if m.status in ("stale", "unsure") else m
        for m in base
    ]
    ranking, reference = rank_members(base), rank_members(relabeled)
    assert ranking.feature_terms == reference.feature_terms == 5
    scores = {item.document_id: item.log_odds for item in reference.items}
    assert {item.document_id: item.log_odds for item in ranking.items} == {
        key: scores[key] for key in ("pending", "stale")
    }
    assert [item.document_id for item in ranking.unsure_items] == ["unsure"]
    assert ranking.unsure_items[0].log_odds == scores["unsure"]
    assert [item.status for item in ranking.items] == ["unscreened", "stale"]


@pytest.mark.parametrize(
    "statuses",
    [
        ("unscreened", "unscreened", "unscreened"),
        ("include", "include", "unscreened"),
        ("exclude", "unsure", "stale"),
    ],
)
def test_cold_start_keeps_collection_order_without_fabricated_scores(
    statuses: tuple[str, str, str],
) -> None:
    members = [
        member(identifier, status, f"Grounded citation {identifier}")
        for identifier, status in zip(("c", "a", "b"), statuses, strict=True)
    ]
    ranking = rank_members(members)
    assert ranking.model_status == "needs_include_and_exclude"
    assert ranking.feature_terms == 0
    pending = [m.document_id for m in members if m.status in ("unscreened", "stale")]
    assert [item.document_id for item in ranking.items] == pending
    for item in (*ranking.items, *ranking.unsure_items):
        assert item.score is None and item.log_odds is None and item.top_terms == ()
    empty = rank_members([])
    assert empty.model_status == "needs_include_and_exclude"
    assert empty.items == empty.unsure_items == ()


def test_labeled_papers_without_usable_terms_do_not_invent_a_model() -> None:
    ranking = rank_members(
        [
            member("include", "include", "The of 2024", "and is 12"),
            member("exclude", "exclude", "A", "x y z"),
            member("pending", "unscreened", "Grounded citation"),
        ]
    )
    assert ranking.model_status == "no_training_terms" and ranking.feature_terms == 0
    assert ranking.items[0].score is None


def test_public_model_rejects_inconsistent_or_fabricated_fields() -> None:
    database_free = ScreeningPriority.model_validate(
        {
            "collection_id": "col_" + "0" * 32,
            "collection_name": "Synthetic",
            "collection_revision": 1,
            "model_status": "needs_include_and_exclude",
            "ordering": "collection_order",
            "total_documents": 1,
            "counts": {"unscreened": 1, "include": 0, "exclude": 0, "unsure": 0, "stale": 0},
            "training_documents": 0,
            "training_relevant": 0,
            "training_irrelevant": 0,
            "feature_terms": 0,
            "items": [
                {
                    "rank": 1,
                    "document_id": "only",
                    "status": "unscreened",
                    "title": "Synthetic",
                    "title_truncated": False,
                    "score": None,
                    "log_odds": None,
                    "top_terms": [],
                }
            ],
            "unsure_items": [],
        }
    )
    payload = database_free.model_dump()
    for update in (
        {"ordering": "most_likely_relevant_first"},
        {"training_relevant": 1},
        {"feature_terms": 3},
        {"total_documents": 2},
        {"items": [{**payload["items"][0], "score": 0.9, "log_odds": 2.0}]},
        {"items": [{**payload["items"][0], "rank": 2}]},
        {"items": [{**payload["items"][0], "status": "unsure"}]},
        {"model_status": "ready"},
    ):
        with pytest.raises(ValidationError):
            ScreeningPriority.model_validate({**payload, **update})


@pytest.fixture
def store(tmp_path: Path) -> tuple[SQLitePaperScreening, Path]:
    database = tmp_path / "priority.sqlite3"
    SQLiteDocumentStore(database).add_documents(
        [
            Document(document_id=identifier, title=title, text=text, source="synthetic:test")
            for identifier, title, text in (
                ("rel", "Citation grounding for scientific answers", "Evidence passages."),
                ("irr", "Battery cathode coatings", "Lithium cells."),
                ("cand-rel", "Grounded evidence citation", "Answers cite passages."),
                ("cand-irr", "Cathode battery fade", "Cells degrade." + " " * 4000 + "citation"),
                ("unsure", "Evidence and batteries", "Mixed."),
                ("long", "Title " + "x" * 400, "Unrelated text."),
            )
        ],
        [],
    )
    return SQLitePaperScreening(database), database


def decide(
    screening: SQLitePaperScreening,
    collection: PaperCollection,
    identifier: str,
    decision: str,
    expected: int = 0,
) -> None:
    screening.submit(
        collection.collection_id,
        identifier,
        ScreeningSubmission.model_validate(
            {
                "collection_revision": collection.revision,
                "expected_decision_revision": expected,
                "decision": decision,
                "reason": "Synthetic human opinion for a test.",
            }
        ),
    )


def test_sqlite_reader_uses_one_snapshot_and_never_writes(
    store: tuple[SQLitePaperScreening, Path],
) -> None:
    screening, database = store
    ids = ("long", "cand-irr", "rel", "irr", "cand-rel", "unsure")
    collection = screening.collections.create(name="Synthetic priority", document_ids=ids)
    cold = prioritize_screening(screening, collection.collection_id, collection_revision=1)
    assert cold.model_status == "needs_include_and_exclude"
    assert [item.document_id for item in cold.items] == list(ids)
    for identifier, decision in (("rel", "include"), ("irr", "exclude"), ("unsure", "unsure")):
        decide(screening, collection, identifier, decision)
    with closing(sqlite3.connect(database)) as connection:
        before_rows = connection.execute(
            "SELECT * FROM paper_screening_decisions ORDER BY document_id"
        ).fetchall()
    before = database.read_bytes()
    result = prioritize_screening(screening, collection.collection_id, collection_revision=1)
    assert database.read_bytes() == before
    with closing(sqlite3.connect(database)) as connection:
        assert (
            connection.execute(
                "SELECT * FROM paper_screening_decisions ORDER BY document_id"
            ).fetchall()
            == before_rows
        )
    assert result.model_status == "ready" and result.ordering == "most_likely_relevant_first"
    assert result.collection_name == "Synthetic priority" and result.total_documents == 6
    assert result.counts.model_dump() == {
        "unscreened": 3,
        "include": 1,
        "exclude": 1,
        "unsure": 1,
        "stale": 0,
    }
    assert (result.training_documents, result.training_relevant) == (2, 1)
    assert result.items[0].document_id == "cand-rel"
    assert [item.document_id for item in result.unsure_items] == ["unsure"]
    long = next(item for item in result.items if item.document_id == "long")
    assert long.title_truncated and len(long.title) == 300
    candidate = next(item for item in result.items if item.document_id == "cand-irr")
    # "citation" appears only after the 4000-character text prefix, so it is never read.
    assert "citation" not in [term.term for term in candidate.top_terms]
    assert json.loads(result.model_dump_json())["notes"][0].startswith("Ordering only")


def test_collection_revision_change_makes_old_decisions_stale_not_labels(
    store: tuple[SQLitePaperScreening, Path],
) -> None:
    screening, _ = store
    collection = screening.collections.create(name="Stale", document_ids=["rel", "irr", "cand-rel"])
    decide(screening, collection, "rel", "include")
    decide(screening, collection, "irr", "exclude")
    assert prioritize_screening(screening, collection.collection_id, collection_revision=1)
    renamed = screening.collections.replace(
        collection.collection_id,
        name="Stale renamed",
        document_ids=collection.document_ids,
        expected_revision=1,
    )
    result = prioritize_screening(screening, renamed.collection_id, collection_revision=2)
    assert result.model_status == "needs_include_and_exclude"
    assert [(item.document_id, item.status) for item in result.items] == [
        ("rel", "stale"),
        ("irr", "stale"),
        ("cand-rel", "unscreened"),
    ]
    with pytest.raises(CollectionError) as conflict:
        prioritize_screening(screening, collection.collection_id, collection_revision=1)
    assert conflict.value.code == "collection_revision_conflict"
    with pytest.raises(CollectionError) as missing:
        prioritize_screening(screening, "col_" + "f" * 32, collection_revision=1)
    assert missing.value.status_code == 404
    with pytest.raises(ValidationError):
        prioritize_screening(screening, "not-a-collection", collection_revision=1)


@pytest.mark.parametrize(
    "corruption",
    [
        "UPDATE documents SET title = 'bad' || char(0) || 'title' WHERE document_id = 'rel'",
        "UPDATE documents SET text = CAST('text' AS BLOB) WHERE document_id = 'rel'",
        "UPDATE documents SET text = X'C328' WHERE document_id = 'rel'",
    ],
)
def test_invalid_stored_text_fails_without_repair(
    store: tuple[SQLitePaperScreening, Path], corruption: str
) -> None:
    screening, database = store
    collection = screening.collections.create(name="Corrupt", document_ids=["rel", "irr"])
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(corruption)
    with pytest.raises(CollectionError) as error:
        prioritize_screening(screening, collection.collection_id, collection_revision=1)
    assert (error.value.code, error.value.status_code) == ("invalid_document_record", 409)


def test_oversized_ordering_fails_without_partial_result(
    store: tuple[SQLitePaperScreening, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    screening, _ = store
    collection = screening.collections.create(name="Large", document_ids=["rel", "irr"])
    monkeypatch.setattr(screening_priority, "MAX_PRIORITY_RESPONSE_BYTES", 100)
    with pytest.raises(CollectionError) as error:
        prioritize_screening(screening, collection.collection_id, collection_revision=1)
    assert (error.value.code, error.value.status_code) == ("screening_priority_too_large", 413)


def test_inconsistent_snapshots_fail_as_invalid_priority(
    store: tuple[SQLitePaperScreening, Path],
) -> None:
    screening, _ = store
    collection = screening.collections.create(name="Guard", document_ids=["rel", "irr"])
    queue = screening.list_queue(collection.collection_id, collection_revision=1, limit=100)
    members = [member(item.document_id, item.status, "Synthetic") for item in queue.items]
    assert build_priority(collection, queue, members).model_status == "needs_include_and_exclude"
    for invalid in (
        list(reversed(members)),
        [member("rel", "include", "Synthetic"), members[1]],
        [member("rel", "unscreened", "x" * 301), members[1]],
    ):
        with pytest.raises(CollectionError) as error:
            build_priority(collection, queue, invalid)
        assert (error.value.code, error.value.status_code) == ("invalid_screening_priority", 409)


def paper(identifier: str, relevant: bool, title: str) -> SimulatedPaper:
    return SimulatedPaper(document_id=identifier, title=title, text="", relevant=relevant)


def test_simulation_metrics_follow_documented_formulas() -> None:
    papers = [
        paper("p1", False, "Battery cathode"),
        paper("p2", True, "Grounded citation evidence"),
        paper("p3", False, "Battery fade"),
        paper("p4", False, "Cathode cells"),
        paper("p5", True, "Grounded citation answers"),
    ]
    default = simulate_screening(papers, strategy="collection_order", recall_target=0.5)
    assert default.screened_document_ids == ("p1", "p2", "p3", "p4", "p5")
    assert default.relevant_found == (0, 1, 1, 1, 2)
    assert (default.relevant_target, default.papers_to_recall_target) == (1, 2)
    assert default.papers_to_find_all_relevant == 5
    assert default.work_saved_over_sampling == pytest.approx((5 - 2) / 5 - 0.5)
    assert default.longest_irrelevant_run_before_last_relevant == 2
    prioritized = simulate_screening(papers, strategy="most_likely_relevant_first")
    # Cold start uses saved order until p1 (exclude) and p2 (include) are labeled.
    assert prioritized.screened_document_ids[:3] == ("p1", "p2", "p5")
    assert prioritized.papers_to_find_all_relevant == 3
    assert prioritized == simulate_screening(papers, strategy="most_likely_relevant_first")


@pytest.mark.parametrize(
    "papers,options",
    [
        ([paper("a", False, "x"), paper("b", False, "y")], {}),
        ([paper("a", True, "x"), paper("a", False, "y")], {}),
        ([paper("a", True, "x")], {}),
        ([paper(f"p{i}", True, "x") for i in range(101)], {}),
        ([paper("a", True, "x"), paper("b", False, "y")], {"recall_target": 0.0}),
        ([paper("a", True, "x"), paper("b", False, "y")], {"recall_target": float("nan")}),
        ([paper("a", True, "x"), paper("b", False, "y")], {"strategy": "random"}),
    ],
)
def test_simulation_rejects_invalid_inputs(
    papers: list[SimulatedPaper], options: dict[str, object]
) -> None:
    arguments: dict[str, object] = {"strategy": "collection_order", **options}
    with pytest.raises(ValueError):
        simulate_screening(papers, **arguments)  # type: ignore[arg-type]


def test_prioritized_order_needs_fewer_papers_on_the_synthetic_set() -> None:
    papers = synthetic_papers()
    for seed in range(10):
        ordered = collection_order(papers, seed)
        fast = simulate_screening(ordered, strategy="most_likely_relevant_first")
        slow = simulate_screening(ordered, strategy="collection_order")
        assert fast.relevant_target == slow.relevant_target == 19
        assert fast.papers_to_recall_target < slow.papers_to_recall_target
        assert fast.work_saved_over_sampling > slow.work_saved_over_sampling
        assert sorted(fast.screened_document_ids) == sorted(p.document_id for p in papers)
        assert fast.relevant_found[-1] == slow.relevant_found[-1] == 20
