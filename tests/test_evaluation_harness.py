"""Tests for EvaluationHarness."""

import pytest

from evaluation.harness import EvalCase, EvaluationHarness
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


def test_docstring_mentions_frontier_models() -> None:
    doc = EvaluationHarness.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
