"""Tests for TimeDecayGate postprocessor."""

from datetime import UTC, date, datetime

import pytest

from retrieval.models import Chunk, SearchResult
from retrieval.time_decay_gate import TimeDecayGate


def _result(
    chunk_id: str,
    score: float = 1.0,
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text="body",
            source="test",
            metadata=metadata or {},
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_rejects_non_positive_half_life() -> None:
    with pytest.raises(ValueError, match="half_life_days"):
        TimeDecayGate(half_life_days=0)


def test_rejects_non_finite_half_life() -> None:
    with pytest.raises(ValueError, match="half_life_days"):
        TimeDecayGate(half_life_days=float("nan"))


def test_empty_results() -> None:
    assert TimeDecayGate(as_of=date(2026, 1, 1)).gate([]) == []


def test_top_k_zero_returns_empty() -> None:
    gate = TimeDecayGate(as_of=date(2026, 1, 1))
    assert gate.gate([_result("a", metadata={"year": "2025"})], top_k=0) == []


def test_decays_older_documents_more() -> None:
    gate = TimeDecayGate(half_life_days=365.0, as_of=date(2026, 1, 1))
    hits = [
        _result("old", 1.0, {"year": "2020"}),
        _result("new", 1.0, {"published_at": "2025-12-01"}),
    ]
    kept = gate.gate(hits)
    assert [r.chunk.chunk_id for r in kept] == ["new", "old"]
    assert kept[0].score > kept[1].score


def test_half_life_halves_score_after_one_period() -> None:
    gate = TimeDecayGate(half_life_days=365.0, as_of=date(2026, 1, 1))
    kept = gate.gate([_result("a", 1.0, {"published_at": "2025-01-01"})])
    assert kept[0].score == pytest.approx(0.5, rel=1e-3)


def test_missing_date_gets_zero_decay() -> None:
    gate = TimeDecayGate(as_of=date(2026, 1, 1))
    kept = gate.gate([_result("missing", 0.9), _result("ok", 0.5, {"year": "2025"})])
    by_id = {r.chunk.chunk_id: r.score for r in kept}
    assert by_id["missing"] == 0.0
    assert by_id["ok"] > 0.0


def test_prefers_published_at_over_year() -> None:
    gate = TimeDecayGate(half_life_days=365.0, as_of=date(2026, 1, 1))
    kept = gate.gate(
        [
            _result(
                "a",
                1.0,
                {"published_at": "2025-01-01", "year": "2020"},
            )
        ]
    )
    assert kept[0].score == pytest.approx(0.5, rel=1e-3)


def test_provenance_rewritten() -> None:
    gate = TimeDecayGate(as_of=date(2026, 1, 1))
    kept = gate.gate([_result("a", metadata={"year": "2025"})])
    assert kept[0].retriever == "time_decay_gate"
    assert kept[0].path == ["hybrid", "bm25"]


def test_top_k_limits_output() -> None:
    gate = TimeDecayGate(as_of=datetime(2026, 1, 1, tzinfo=UTC))
    hits = [
        _result("a", 1.0, {"year": "2025"}),
        _result("b", 1.0, {"year": "2024"}),
        _result("c", 1.0, {"year": "2023"}),
    ]
    kept = gate.gate(hits, top_k=2)
    assert len(kept) == 2


def test_does_not_mutate_inputs() -> None:
    original = [_result("a", 1.0, {"year": "2025"})]
    path_before = list(original[0].path)
    score_before = original[0].score
    TimeDecayGate(as_of=date(2026, 1, 1)).gate(original)
    assert original[0].path == path_before
    assert original[0].retriever == "bm25"
    assert original[0].score == score_before


def test_docstring_mentions_frontier_models() -> None:
    assert "GPT-5.5" in (TimeDecayGate.__doc__ or "")
