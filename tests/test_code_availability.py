"""Tests for CodeAvailabilityBooster."""

import pytest

from retrieval.code_availability import CodeAvailabilityBooster
from retrieval.models import Chunk, SearchResult


def _result(
    chunk_id: str,
    score: float,
    *,
    title: str = "",
    text: str = "",
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=title or chunk_id,
            text=text or f"text for {chunk_id}",
            source="test",
            metadata={} if metadata is None else metadata,
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        CodeAvailabilityBooster(alpha=alpha)


def test_empty_results_ok() -> None:
    assert CodeAvailabilityBooster().boost([]) == []


def test_boosts_github_mention_over_closed_code() -> None:
    closed = _result("closed", score=1.0, text="A methods paper with no repository.")
    opened = _result(
        "open",
        score=0.0,
        text="Code is at https://github.com/org/repo for reproducibility.",
    )
    boosted = CodeAvailabilityBooster(alpha=1.0).boost([closed, opened])

    assert [row.title for row in boosted] == ["open", "closed"]
    assert boosted[0].score == pytest.approx(1.0)
    assert boosted[0].code_signal == 1.0
    assert any("github.com" in reason for reason in boosted[0].reasons)
    assert boosted[1].code_signal == 0.0
    assert boosted[1].reasons == ("no code availability signal",)
    assert closed.score == 1.0  # originals not mutated


def test_detects_phrase_cues_in_title_and_abstract() -> None:
    rows = [
        {"title": "Survey", "abstract": "We release source code.", "score": 0.1},
        {"title": "With code available online", "abstract": "No URL.", "score": 0.1},
        {"title": "Baseline", "abstract": "Our implementation outperforms priors.", "score": 0.1},
        {"title": "GitLab mirror", "abstract": "See gitlab.com/org/proj.", "score": 0.1},
        {"title": "Bitbucket", "abstract": "Artifacts on bitbucket.org/team/repo.", "score": 0.1},
    ]
    boosted = CodeAvailabilityBooster(alpha=1.0).boost(rows)
    assert all(row.code_signal == 1.0 for row in boosted)
    joined = " ".join(reason for row in boosted for reason in row.reasons)
    assert "source code" in joined
    assert "code available" in joined
    assert "implementation" in joined
    assert "gitlab" in joined
    assert "bitbucket" in joined


def test_formula_matches_one_minus_alpha_old_plus_alpha_signal() -> None:
    result = _result(
        "only",
        score=0.8,
        metadata={"abstract": "Official implementation on github.com/acme/tool."},
    )
    boosted = CodeAvailabilityBooster(alpha=0.3).boost([result])
    assert boosted[0].score == pytest.approx((1 - 0.3) * 0.8 + 0.3 * 1.0)
    assert boosted[0].prior_score == pytest.approx(0.8)
    assert boosted[0].result is not None
    assert boosted[0].result.retriever == "code_availability"
    assert boosted[0].result.path == ["hybrid", "bm25"]


def test_stable_ordering_for_tied_scores() -> None:
    first = {"title": "A", "abstract": "github.com/a", "score": 0.5}
    second = {"title": "B", "abstract": "github.com/b", "score": 0.5}
    boosted = CodeAvailabilityBooster(alpha=0.0).boost([first, second])
    assert [row.title for row in boosted] == ["A", "B"]


def test_top_k_truncates_after_boost() -> None:
    rows = [
        {"title": "no", "abstract": "plain text", "score": 0.9},
        {"title": "yes", "abstract": "code available at gitlab.com/x", "score": 0.1},
        {"title": "also", "abstract": "source code released", "score": 0.2},
    ]
    boosted = CodeAvailabilityBooster(alpha=1.0).boost(rows, top_k=2)
    assert len(boosted) == 2
    assert all(row.code_signal == 1.0 for row in boosted)


def test_reads_abstract_metadata_on_search_result() -> None:
    result = _result(
        "meta",
        score=0.0,
        text="body without hosts",
        metadata={"abstract": "Mirror on bitbucket.org/team/repo"},
    )
    boosted = CodeAvailabilityBooster(alpha=1.0).boost([result])
    assert boosted[0].code_signal == 1.0


def test_docstring_mentions_frontier_models_and_gap() -> None:
    doc = CodeAvailabilityBooster.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
    assert "PapersWithCode" in doc or "Semantic Scholar" in doc
    assert "OpenAccessPreferencer" in doc
