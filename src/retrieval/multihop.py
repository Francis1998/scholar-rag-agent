"""Bounded multi-hop entity-chain retrieval."""

from retrieval.models import SearchResult
from storage.graph_store import SQLiteGraphStore


class MultiHopRetriever:
    """Follow entity relationship chains with depth and candidate bounds."""

    def __init__(self, graph_store: SQLiteGraphStore, max_depth: int = 3) -> None:
        """Create a multi-hop graph retriever."""
        self._graph_store = graph_store
        self._max_depth = min(max_depth, 5)

    async def retrieve(
        self,
        query: str,
        seed_entities: list[str],
        depth: int = 3,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Retrieve chunks by traversing entity chains up to a bounded depth."""
        del query
        bounded_depth = min(depth, self._max_depth, 5)
        if bounded_depth <= 0:
            return []
        frontier = {entity.lower() for entity in seed_entities}
        visited = set(frontier)
        results: dict[str, SearchResult] = {}
        for hop in range(1, bounded_depth + 1):
            if not frontier or len(results) >= limit:
                break
            chunks = self._graph_store.chunks_for_entities(sorted(frontier), limit=limit)
            for chunk in chunks:
                results.setdefault(
                    chunk.chunk_id,
                    SearchResult(
                        chunk=chunk, score=1.0 / hop, retriever="multihop", path=sorted(frontier)
                    ),
                )
            neighbours = self._graph_store.neighbours(sorted(frontier), limit=limit * 4)
            frontier = {entity.lower() for entity in neighbours if entity.lower() not in visited}
            visited.update(frontier)
        return sorted(results.values(), key=lambda result: result.score, reverse=True)[:limit]
