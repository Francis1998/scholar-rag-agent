"""Reciprocal rank fusion utilities."""

from retrieval.models import SearchResult


def reciprocal_rank_fusion(
    result_sets: list[list[SearchResult]],
    limit: int = 10,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Fuse ranked result sets using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    best_results: dict[str, SearchResult] = {}
    paths: dict[str, list[str]] = {}
    for results in result_sets:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
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
