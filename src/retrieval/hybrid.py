"""Hybrid dense and sparse retrieval with HyDE and RRF."""

from retrieval.dense import DenseRetriever
from retrieval.hyde import HyDEExpander
from retrieval.mmr import MMRDiversifier
from retrieval.models import Chunk, SearchResult
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.sparse import BM25Retriever


class HybridRetriever:
    """Combine dense and BM25 search using HyDE and RRF."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        hyde_expander: HyDEExpander,
        diversifier: MMRDiversifier | None = None,
    ) -> None:
        """Create a hybrid retriever from dense, sparse, and HyDE components.

        Args:
            dense_retriever: Dense vector retriever.
            sparse_retriever: BM25 lexical retriever.
            hyde_expander: HyDE query expander.
            diversifier: Optional MMR diversifier applied to fused results to
                reduce near-duplicate chunks. When ``None`` (default), fused
                results are returned in RRF order unchanged.
        """
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._hyde_expander = hyde_expander
        self._diversifier = diversifier

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks into both dense and sparse retrievers."""
        self._dense_retriever.add_chunks(chunks)
        self._sparse_retriever.add_chunks(chunks)

    async def retrieve(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Retrieve fused dense and sparse results for a query."""
        expanded_query = await self._hyde_expander.expand(query)
        dense_results = await self._dense_retriever.retrieve(expanded_query, limit=limit)
        sparse_results = await self._sparse_retriever.retrieve(expanded_query, limit=limit)
        fused_results = reciprocal_rank_fusion([dense_results, sparse_results], limit=limit)
        if self._diversifier is not None:
            return self._diversifier.diversify(fused_results, top_k=limit)
        return fused_results
