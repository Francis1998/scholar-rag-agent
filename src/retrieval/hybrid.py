"""Hybrid dense and sparse retrieval with HyDE and RRF."""

from retrieval.dense import DenseRetriever
from retrieval.hyde import HyDEExpander
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
    ) -> None:
        """Create a hybrid retriever from dense, sparse, and HyDE components."""
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._hyde_expander = hyde_expander

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks into both dense and sparse retrievers."""
        self._dense_retriever.add_chunks(chunks)
        self._sparse_retriever.add_chunks(chunks)

    async def retrieve(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Retrieve fused dense and sparse results for a query."""
        expanded_query = await self._hyde_expander.expand(query)
        dense_results = await self._dense_retriever.retrieve(expanded_query, limit=limit)
        sparse_results = await self._sparse_retriever.retrieve(expanded_query, limit=limit)
        return reciprocal_rank_fusion([dense_results, sparse_results], limit=limit)
