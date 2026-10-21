"""Adaptive cross-encoder-style reranking."""

from retrieval.models import SearchResult
from retrieval.sparse import tokenize


class AdaptiveReranker:
    """Rerank chunks lexically by default or with an explicitly requested cross-encoder."""

    def __init__(
        self,
        use_cross_encoder: bool = False,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """Create a reranker; opt-in dependency or model initialization errors propagate."""
        self._model = None
        if use_cross_encoder:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return reranked results with adaptive scores, skipping inference for empty input."""
        if not results:
            return []
        if self._model is not None:
            pairs = [(query, result.chunk.text) for result in results]
            raw_scores = self._model.predict(pairs)
            return sorted(
                [
                    SearchResult(
                        chunk=result.chunk,
                        score=float(score),
                        retriever="cross_encoder",
                        path=[*result.path, result.retriever],
                    )
                    for result, score in zip(results, raw_scores, strict=True)
                ],
                key=lambda result: result.score,
                reverse=True,
            )
        query_terms = set(tokenize(query))
        reranked = []
        for result in results:
            chunk_terms = set(tokenize(result.chunk.text))
            overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            reranked.append(
                SearchResult(
                    chunk=result.chunk,
                    score=result.score + overlap,
                    retriever="lexical_rerank",
                    path=[*result.path, result.retriever],
                )
            )
        return sorted(reranked, key=lambda result: result.score, reverse=True)
