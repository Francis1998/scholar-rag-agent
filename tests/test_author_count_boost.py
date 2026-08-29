"""Tests for author-count boost postprocessing."""

import pytest

from retrieval.author_count_boost import AuthorCountBooster
from retrieval.models import Chunk, SearchResult


def _result(chunk_id: str, score: float, metadata: dict[str, str]) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            title=chunk_id,
            text=f"text for {chunk_id}",
            source="test",
            metadata=metadata,
        ),
        score=score,
        retriever="bm25",
        path=["hybrid"],
    )


def test_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="author bounds"):
        AuthorCountBooster(min_authors=5, max_authors=2)


def test_prefers_mid_author_band() -> None:
    mid = _result("mid", 0.0, {"author_count": "4"})
    solo = _result("solo", 1.0, {"author_count": "1"})
    boosted = AuthorCountBooster(alpha=1.0).boost([solo, mid])
    assert [item.chunk.chunk_id for item in boosted] == ["mid", "solo"]
    assert boosted[0].retriever == "author_count_boost"


def test_parses_authors_string() -> None:
    result = _result("a", 0.0, {"authors": "Ada Lovelace and Alan Turing"})
    boosted = AuthorCountBooster(alpha=1.0, min_authors=2, max_authors=2).boost([result])
    assert boosted[0].score == pytest.approx(1.0)


def test_missing_author_metadata_demotes() -> None:
    result = _result("a", 0.0, {})
    boosted = AuthorCountBooster(alpha=1.0).boost([result])
    assert boosted[0].score == pytest.approx(0.2)


def test_top_k() -> None:
    rows = [_result("a", 0.0, {"author_count": "3"}), _result("b", 0.0, {"author_count": "1"})]
    assert len(AuthorCountBooster(alpha=1.0).boost(rows, top_k=1)) == 1
