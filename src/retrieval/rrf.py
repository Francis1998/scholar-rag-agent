"""Reciprocal rank fusion utilities."""

from retrieval.models import SearchResult


def reciprocal_rank_fusion(
    result_sets: list[list[SearchResult]],
    limit: int | None = 10,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Fuse rankings with one vote per exact chunk ID per input list.

    Duplicates use their first original 1-based rank in each list. A ``None``
    limit returns all results; integer limits retain Python slicing semantics.
    """
    scores: dict[str, float] = {}
    best_results: dict[str, SearchResult] = {}
    paths: dict[str, list[str]] = {}
    for results in result_sets:
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + rank)
            best_results.setdefault(chunk_id, result)
            paths.setdefault(chunk_id, []).append(result.retriever)
    fused = [
        SearchResult(
            chunk=best_results[chunk_id].chunk,
            score=score,
            retriever="rrf",
            path=paths[chunk_id],
        )
        for chunk_id, score in scores.items()
    ]
    return sorted(fused, key=lambda result: result.score, reverse=True)[:limit]
