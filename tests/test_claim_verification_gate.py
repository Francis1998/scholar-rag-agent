"""Tests for deterministic claim verification groundedness gating."""

import pytest

from retrieval.claim_verification_gate import ClaimVerificationGate
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, text: str, title: str = "Study") -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=title,
            text=text,
            source="test",
        ),
        score=0.8,
        retriever="rrf",
    )


def test_marks_supported_and_unsupported_claims_with_groundedness() -> None:
    results = [
        _result(
            "mol",
            "Graph neural networks improve molecular property prediction accuracy.",
        )
    ]
    answer = (
        "Graph neural networks improve molecular property prediction. The moon is made of cheese."
    )

    report = ClaimVerificationGate().verify(answer, results)

    assert len(report.claims) == 2
    assert report.claims[0].supported is True
    assert report.claims[0].support_score == 1.0
    assert report.claims[0].supporting_chunk_ids == ("mol",)
    assert report.claims[1].supported is False
    assert report.claims[1].supporting_chunk_ids == ()
    assert report.supported_count == 1
    assert report.unsupported_count == 1
    assert report.groundedness == pytest.approx(0.5)


def test_empty_answer_or_evidence_yields_zero_groundedness() -> None:
    gate = ClaimVerificationGate()

    empty_answer = gate.verify("", [_result("a", "Graph retrieval works.")])
    no_evidence = gate.verify(
        "Graph neural networks improve retrieval.",
        [],
    )

    assert empty_answer.claims == ()
    assert empty_answer.groundedness == 0.0
    assert no_evidence.claims[0].supported is False
    assert no_evidence.groundedness == 0.0
    assert no_evidence.unsupported_count == 1


def test_stopword_only_claim_is_unsupported() -> None:
    report = ClaimVerificationGate().verify(
        "The and of.",
        [_result("paper", "Relevant scientific evidence.")],
    )

    assert len(report.claims) == 1
    assert report.claims[0].supported is False
    assert report.claims[0].support_score == 0.0
    assert report.groundedness == 0.0


def test_support_threshold_controls_verdict() -> None:
    results = [_result("partial", "Graph retrieval baselines were evaluated.")]
    answer = "Graph neural retrieval molecular prediction."

    lenient = ClaimVerificationGate(support_threshold=0.4).verify(answer, results)
    strict = ClaimVerificationGate(support_threshold=0.8).verify(answer, results)

    assert lenient.claims[0].support_score == pytest.approx(0.4)
    assert lenient.claims[0].supported is True
    assert strict.claims[0].supported is False


def test_optional_llm_stub_is_accepted_but_unused() -> None:
    class _UnusedLLM:
        async def generate(self, request: object) -> object:  # pragma: no cover
            raise AssertionError("LLM must not be called")

    report = ClaimVerificationGate(llm=_UnusedLLM()).verify(  # type: ignore[arg-type]
        "Graph retrieval improves ranking.",
        [_result("hit", "Graph retrieval improves ranking quality.")],
    )

    assert report.supported_count == 1
    assert report.groundedness == 1.0


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_support_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="support_threshold"):
        ClaimVerificationGate(support_threshold=threshold)
