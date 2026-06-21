"""Adaptive cross-encoder-style reranking."""

from retrieval.models import SearchResult
from retrieval.sparse import tokenize


class AdaptiveReranker:
    """Rerank chunks using a cross-encoder when available and lexical fallback otherwise."""

    def __init__(
        self,
        use_cross_encoder: bool = False,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """Create an adaptive reranker."""
        self._model = None
        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(model_name)
            except Exception:
                self._model = None

    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return reranked results with adaptive scores."""
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
