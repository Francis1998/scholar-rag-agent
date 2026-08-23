"""Tests for deterministic citation-marker groundedness scoring."""

import pytest

from retrieval.citation_groundedness_score import CitationGroundednessScorer
from retrieval.models import Chunk, SearchResult


def _result(
    chunk_id: str,
    text: str,
    title: str = "Study",
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=title,
            text=text,
            source="test",
            metadata=metadata or {},
        ),
        score=0.8,
        retriever="rrf",
    )


def test_bracket_citation_resolves_by_result_index() -> None:
    results = [
        _result("mol", "Graph neural networks improve molecular property prediction."),
        _result("clinical", "A clinical trial measured drug efficacy outcomes."),
    ]
    answer = (
        "Graph neural networks improve molecular property prediction [1]. "
        "This sentence cites unrelated evidence [2]."
    )

    report = CitationGroundednessScorer().score(answer, results)

    assert len(report.citations) == 2
    assert report.citations[0].marker == "[1]"
    assert report.citations[0].candidate_chunk_ids == ("mol",)
    assert report.citations[0].grounded is True
    assert report.citations[0].overlap_score == pytest.approx(1.0)
    assert report.citations[1].marker == "[2]"
    assert report.citations[1].grounded is False
    assert report.grounded_count == 1
    assert report.ungrounded_count == 1
    assert report.groundedness == pytest.approx(0.5)


def test_bracket_citation_with_multiple_indices_uses_best_overlap() -> None:
    results = [
        _result("weak", "Unrelated background text."),
        _result("strong", "Graph retrieval improves ranking quality substantially."),
    ]
    answer = "Graph retrieval improves ranking quality [1, 2]."

    report = CitationGroundednessScorer().score(answer, results)

    assert report.citations[0].candidate_chunk_ids == ("weak", "strong")
    assert report.citations[0].grounded is True
    assert report.citations[0].overlap_score == pytest.approx(1.0)


def test_out_of_range_bracket_index_is_unresolved_and_ungrounded() -> None:
    results = [_result("only", "Graph retrieval improves ranking.")]
    answer = "Graph retrieval improves ranking [5]."

    report = CitationGroundednessScorer().score(answer, results)

    assert report.citations[0].candidate_chunk_ids == ()
    assert report.citations[0].grounded is False
    assert report.groundedness == 0.0


def test_author_year_citation_resolves_via_metadata() -> None:
    results = [
        _result(
            "s1",
            "Graph neural networks improve molecular property prediction.",
            metadata={"authors": "John Smith, Alice Lee", "year": "2020"},
        ),
        _result("s2", "Unrelated evidence about clinical trials.", metadata={"year": "2019"}),
    ]
    answer = "Graph neural networks improve molecular property prediction (Smith, 2020)."

    report = CitationGroundednessScorer().score(answer, results)

    assert report.citations[0].marker == "(Smith, 2020)"
    assert report.citations[0].candidate_chunk_ids == ("s1",)
    assert report.citations[0].grounded is True


@pytest.mark.parametrize(
    "marker",
    ["(Smith et al., 2020)", "(Smith and Jones, 2020)", "(Smith & Jones, 2020)"],
)
def test_author_year_citation_supports_multi_author_forms(marker: str) -> None:
    results = [
        _result(
            "s1",
            "Graph neural networks improve molecular property prediction.",
            metadata={"authors": "John Smith, Bob Jones", "year": "2020"},
        )
    ]
    answer = f"Graph neural networks improve molecular property prediction {marker}."

    report = CitationGroundednessScorer().score(answer, results)

    assert report.citations[0].candidate_chunk_ids == ("s1",)
    assert report.citations[0].grounded is True


def test_author_year_citation_requires_matching_year() -> None:
    results = [
        _result(
            "s1",
            "Graph neural networks improve molecular property prediction.",
            metadata={"authors": "John Smith", "year": "2015"},
        )
    ]
    answer = "Graph neural networks improve molecular property prediction (Smith, 2020)."

    report = CitationGroundednessScorer().score(answer, results)

    assert report.citations[0].candidate_chunk_ids == ()
    assert report.citations[0].grounded is False


def test_sentence_without_citation_markers_is_excluded() -> None:
    results = [_result("mol", "Graph neural networks improve molecular property prediction.")]
    answer = "Graph neural networks improve molecular property prediction [1]. No citation here."

    report = CitationGroundednessScorer().score(answer, results)

    assert len(report.citations) == 1
    assert report.citations[0].marker == "[1]"


def test_no_citations_yields_empty_report() -> None:
    report = CitationGroundednessScorer().score(
        "No citations appear in this answer at all.",
        [_result("mol", "Some evidence text.")],
    )

    assert report.citations == ()
    assert report.groundedness == 0.0
    assert report.grounded_count == 0
    assert report.ungrounded_count == 0


def test_stopword_only_sentence_is_ungrounded_even_with_valid_index() -> None:
    results = [_result("mol", "The and of this that.")]
    answer = "The and of [1]."

    report = CitationGroundednessScorer().score(answer, results)

    assert report.citations[0].candidate_chunk_ids == ("mol",)
    assert report.citations[0].grounded is False
    assert report.citations[0].overlap_score == 0.0


def test_overlap_threshold_controls_verdict() -> None:
    results = [_result("partial", "Graph retrieval baselines were evaluated recently.")]
    answer = "Graph neural retrieval molecular prediction results [1]."

    lenient = CitationGroundednessScorer(overlap_threshold=0.2).score(answer, results)
    strict = CitationGroundednessScorer(overlap_threshold=0.9).score(answer, results)

    assert lenient.citations[0].grounded is True
    assert strict.citations[0].grounded is False


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_overlap_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="overlap_threshold"):
        CitationGroundednessScorer(overlap_threshold=threshold)


def test_empty_answer_or_no_results_yields_stable_output() -> None:
    scorer = CitationGroundednessScorer()

    assert scorer.score("", [_result("a", "Graph retrieval works.")]).citations == ()
    assert scorer.score("No markers at all.", []).citations == ()
