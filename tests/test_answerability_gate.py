"""Tests for the deterministic lexical answerability gate."""

import pytest

from retrieval.answerability_gate import AnswerabilityGate
from retrieval.models import Chunk, SearchResult


def _result(
    chunk_id: str,
    text: str,
    score: float = 1.0,
    title: str | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=title if title is not None else chunk_id,
            text=text,
            source="test",
            metadata={},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_score": -0.1}, "min_score"),
        ({"min_score": 1.1}, "min_score"),
        ({"min_score": float("nan")}, "min_score"),
        ({"answerability_threshold": -0.01}, "answerability_threshold"),
        ({"answerability_threshold": 1.5}, "answerability_threshold"),
        ({"answerability_threshold": float("inf")}, "answerability_threshold"),
    ],
)
def test_rejects_invalid_thresholds(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AnswerabilityGate(**kwargs)


def test_score_reports_per_result_and_mean_answerability() -> None:
    gate = AnswerabilityGate(min_score=0.5, answerability_threshold=0.1)
    results = [
        _result("strong", "Graph neural networks predict molecular properties."),
        _result("weak", "Unrelated agricultural soil moisture survey."),
    ]

    report = gate.score("graph neural networks molecular", results)

    assert report.result_scores[0] == pytest.approx(1.0)
    assert report.result_scores[1] == pytest.approx(0.0)
    assert report.answerability == pytest.approx(0.5)
    assert report.kept_count == 1
    assert report.dropped_count == 1


def test_filter_drops_results_below_min_score() -> None:
    gate = AnswerabilityGate(min_score=0.5, answerability_threshold=0.1)
    strong = _result("strong", "Transformer attention improves translation quality.")
    weak = _result("weak", "Weather forecasts for coastal cities.")
    results = [strong, weak]

    filtered = gate.filter("transformer attention translation", results)

    assert filtered == [strong]
    assert filtered[0] is strong


def test_filter_returns_empty_when_aggregate_below_threshold() -> None:
    gate = AnswerabilityGate(min_score=0.0, answerability_threshold=0.8)
    results = [
        _result("partial", "Neural networks classify images."),
        _result("other", "Bayesian networks encode conditional dependence."),
    ]

    # Per-result coverage is non-zero for "networks", but mean answerability
    # stays below 0.8 so the whole batch is refused.
    filtered = gate.filter("neural networks classify images", results)

    assert filtered == []


def test_empty_results_and_stopword_only_query() -> None:
    gate = AnswerabilityGate()

    empty_report = gate.score("graph neural networks", [])
    assert empty_report == AnswerabilityGate().score("graph neural networks", [])
    assert empty_report.answerability == 0.0
    assert empty_report.result_scores == ()
    assert gate.filter("graph neural networks", []) == []

    stopword_results = [_result("any", "The networks are effective.")]
    stopword_report = gate.score("the and of", stopword_results)
    assert stopword_report.answerability == 0.0
    assert stopword_report.result_scores == (0.0,)
    assert stopword_report.kept_count == 0
    assert stopword_report.dropped_count == 1
    assert gate.filter("the and of", stopword_results) == []


def test_title_contributes_to_coverage() -> None:
    gate = AnswerabilityGate(min_score=0.5, answerability_threshold=0.1)
    results = [_result("title-hit", "Unrelated body text.", title="CRISPR gene editing")]

    report = gate.score("CRISPR gene editing", results)

    assert report.result_scores[0] == pytest.approx(1.0)
    assert gate.filter("CRISPR gene editing", results) == results
