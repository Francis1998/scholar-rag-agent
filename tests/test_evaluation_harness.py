"""Tests for EvaluationHarness."""

import pytest

from evaluation.harness import CaseScore, EvalCase, EvaluationHarness, HarnessReport
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, text: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=text,
            source="fixture",
        ),
        score=score,
        retriever="hybrid",
        path=["hybrid"],
    )


def test_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="cases"):
        EvaluationHarness([])


def test_rejects_case_without_relevant_ids() -> None:
    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        EvaluationHarness([EvalCase(case_id="c1", query="q", relevant_chunk_ids=frozenset())])


def test_rejects_non_positive_k() -> None:
    harness = EvaluationHarness(
        [EvalCase(case_id="c1", query="q", relevant_chunk_ids=frozenset({"a"}))]
    )
    with pytest.raises(ValueError, match="k"):
        harness.evaluate(lambda _q, _k: [], k=0)


def test_hit_and_recall_at_k() -> None:
    harness = EvaluationHarness(
        [
            EvalCase(
                case_id="c1",
                query="hybrid retrieval",
                relevant_chunk_ids=frozenset({"a", "b"}),
            )
        ]
    )

    def retrieve(query: str, k: int) -> list[SearchResult]:
        assert query == "hybrid retrieval"
        return [_result("a", "hybrid dense sparse"), _result("c", "unrelated")][:k]

    report = harness.evaluate(retrieve, k=2)
    assert report.mean_hit_at_k == 1.0
    assert report.mean_recall_at_k == pytest.approx(0.5)
    assert report.cases[0].retrieved_chunk_ids == ("a", "c")


def test_miss_has_zero_hit_and_recall() -> None:
    harness = EvaluationHarness(
        [EvalCase(case_id="c1", query="q", relevant_chunk_ids=frozenset({"gold"}))]
    )
    report = harness.evaluate(lambda _q, _k: [_result("other", "nope")], k=1)
    assert report.mean_hit_at_k == 0.0
    assert report.mean_recall_at_k == 0.0


def test_faithfulness_uses_retrieved_evidence() -> None:
    results = [
        _result("a", "Graph retrieval improves multi hop reasoning over papers."),
    ]
    score = EvaluationHarness.faithfulness(
        "Graph retrieval improves multi hop reasoning.",
        results,
    )
    assert score == pytest.approx(1.0)


def test_faithfulness_partial_overlap() -> None:
    results = [_result("a", "dense sparse hybrid retrieval")]
    score = EvaluationHarness.faithfulness(
        "dense sparse lexical expansion methods",
        results,
    )
    # tokens: dense sparse lexical expansion methods -> 5; overlap dense sparse = 2
    assert score == pytest.approx(0.4)


def test_evaluate_with_answers_sets_faithfulness() -> None:
    harness = EvaluationHarness(
        [
            EvalCase(
                case_id="c1",
                query="q",
                relevant_chunk_ids=frozenset({"a"}),
                gold_answer="hybrid retrieval evidence",
            )
        ]
    )
    report = harness.evaluate(
        lambda _q, _k: [_result("a", "hybrid retrieval evidence supports answers")],
        k=1,
        answers={"c1": "hybrid retrieval evidence"},
    )
    assert report.mean_faithfulness == pytest.approx(1.0)
    assert report.cases[0].faithfulness == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param("galaxies planets", 0.0, id="unsupported"),
        pytest.param("", 0.0, id="empty"),
        pytest.param(" \t\n", 0.0, id="whitespace"),
        pytest.param("...!?", 0.0, id="punctuation"),
        pytest.param("the and of", 0.0, id="stopwords"),
        pytest.param("hybrid astronomy", 0.5, id="partially-grounded"),
    ],
)
def test_evaluate_scores_generated_answers_when_gold_matches(answer: str, expected: float) -> None:
    harness = EvaluationHarness(
        [
            EvalCase(
                case_id="c1",
                query="q",
                relevant_chunk_ids=frozenset({"a"}),
                gold_answer="hybrid retrieval",
            )
        ]
    )
    report = harness.evaluate(
        lambda _q, _k: [_result("a", "hybrid retrieval")],
        answers={"c1": answer},
    )

    assert report.cases[0].faithfulness == pytest.approx(expected)
    assert report.mean_faithfulness == pytest.approx(expected)
    assert report.cases[0].reference_coverage == pytest.approx(1.0)
    assert report.mean_reference_coverage == pytest.approx(1.0)


@pytest.mark.parametrize("gold_answer", [None, "", "astronomy", "hybrid", "hybrid retrieval"])
def test_faithfulness_ignores_gold_answer(gold_answer: str | None) -> None:
    results = [_result("a", "hybrid retrieval")]
    answer = "hybrid retrieval astronomy"

    assert EvaluationHarness.faithfulness(answer, results, gold_answer) == pytest.approx(2 / 3)
    assert EvaluationHarness.faithfulness(
        answer, results, gold_answer=gold_answer
    ) == pytest.approx(2 / 3)


def test_faithfulness_uses_unique_content_tokens_from_titles_and_text() -> None:
    results = [_result("Hybrid", "retrieval retrieval")]

    assert EvaluationHarness.faithfulness(
        "The HYBRID hybrid, retrieval; astronomy.", results
    ) == pytest.approx(2 / 3)


def test_faithfulness_without_evidence_is_zero() -> None:
    assert EvaluationHarness.faithfulness("hybrid retrieval", []) == 0.0


@pytest.mark.parametrize("answers", [None, {}, {"other": "hybrid retrieval"}])
def test_missing_answers_leave_faithfulness_unset(answers: dict[str, str] | None) -> None:
    harness = EvaluationHarness(
        [
            EvalCase(
                case_id="c1",
                query="q",
                relevant_chunk_ids=frozenset({"a"}),
                gold_answer="hybrid retrieval",
            )
        ]
    )
    report = harness.evaluate(
        lambda _q, _k: [_result("a", "hybrid retrieval")],
        answers=answers,
    )

    assert report.cases[0].faithfulness is None
    assert report.mean_faithfulness is None
    assert report.cases[0].reference_coverage == pytest.approx(1.0)
    assert report.mean_reference_coverage == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("gold_answer", "expected"),
    [
        (None, None),
        ("", 0.0),
        (" \t\n", 0.0),
        ("...!?", 0.0),
        ("the and of", 0.0),
        ("astronomy", 0.0),
        ("hybrid", 1.0),
        ("HYBRID hybrid astronomy", 0.5),
    ],
)
def test_reference_coverage_is_independent_of_faithfulness(
    gold_answer: str | None, expected: float | None
) -> None:
    harness = EvaluationHarness(
        [
            EvalCase(
                case_id="c1",
                query="q",
                relevant_chunk_ids=frozenset({"a"}),
                gold_answer=gold_answer,
            )
        ]
    )
    report = harness.evaluate(
        lambda _q, _k: [_result("a", "hybrid retrieval")],
        answers={"c1": "retrieval"},
    )

    assert report.cases[0].faithfulness == pytest.approx(1.0)
    assert report.mean_faithfulness == pytest.approx(1.0)
    if expected is None:
        assert report.cases[0].reference_coverage is None
        assert report.mean_reference_coverage is None
    else:
        assert report.cases[0].reference_coverage == pytest.approx(expected)
        assert report.mean_reference_coverage == pytest.approx(expected)


def test_aggregate_scores_exclude_missing_values_but_include_empty_answers() -> None:
    harness = EvaluationHarness(
        [
            EvalCase("missing", "q", frozenset({"a"}), gold_answer="hybrid retrieval"),
            EvalCase("empty", "q", frozenset({"a"}), gold_answer=""),
            EvalCase("partial", "q", frozenset({"a"})),
            EvalCase("faithful", "q", frozenset({"a"}), gold_answer="astronomy"),
        ]
    )
    report = harness.evaluate(
        lambda _q, _k: [_result("a", "hybrid retrieval")],
        answers={"empty": "", "partial": "hybrid astronomy", "faithful": "hybrid retrieval"},
    )

    assert [score.faithfulness for score in report.cases] == [None, 0.0, 0.5, 1.0]
    assert report.mean_faithfulness == pytest.approx(0.5)
    assert [score.reference_coverage for score in report.cases] == [1.0, 0.0, None, 0.0]
    assert report.mean_reference_coverage == pytest.approx(1 / 3)
    assert report.mean_hit_at_k == 1.0
    assert report.mean_recall_at_k == 1.0


def test_evaluate_without_answers_or_references_leaves_both_metrics_unset() -> None:
    harness = EvaluationHarness([EvalCase("c1", "q", frozenset({"a"}))])
    report = harness.evaluate(lambda _q, _k: [_result("a", "hybrid retrieval")])

    assert report.cases[0].faithfulness is None
    assert report.mean_faithfulness is None
    assert report.cases[0].reference_coverage is None
    assert report.mean_reference_coverage is None


def test_evaluate_uses_only_top_k_evidence_in_callback_order() -> None:
    harness = EvaluationHarness(
        [EvalCase("c1", "q", frozenset({"a", "c"}), gold_answer="hybrid retrieval evidence")]
    )

    def retrieve(query: str, k: int) -> list[SearchResult]:
        assert query == "q"
        assert k == 2
        return [
            _result("a", "hybrid", score=0.1),
            _result("b", "retrieval", score=0.9),
            _result("c", "evidence", score=1.0),
        ]

    report = harness.evaluate(retrieve, k=2, answers={"c1": "evidence"})

    assert report.k == 2
    assert report.cases[0].retrieved_chunk_ids == ("a", "b")
    assert report.cases[0].hit_at_k == report.mean_hit_at_k == 1.0
    assert report.cases[0].recall_at_k == report.mean_recall_at_k == 0.5
    assert report.cases[0].faithfulness == report.mean_faithfulness == 0.0
    assert report.cases[0].reference_coverage == pytest.approx(2 / 3)
    assert report.mean_reference_coverage == pytest.approx(2 / 3)


def test_report_dataclasses_preserve_existing_positional_constructors() -> None:
    score = CaseScore("c1", 1.0, 1.0, None, ("a",))
    report = HarnessReport((score,), 1.0, 1.0, None, 5)

    assert score.reference_coverage is None
    assert report.mean_reference_coverage is None
