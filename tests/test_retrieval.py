"""Tests for hybrid retrieval components."""

from ingestion.chunking import TextChunker, stable_id
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Document, SearchResult
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.sparse import BM25Retriever


def build_chunks() -> list:
    """Build deterministic fixture chunks."""
    document = Document(
        document_id=stable_id("rag", "doc"),
        title="RAG Methods",
        text=(
            "Hybrid retrieval combines dense embeddings and BM25 sparse search for scientific RAG."
        ),
        source="fixture",
    )
    return TextChunker(chunk_size=240, overlap=0).chunk(document)


async def test_hybrid_retriever_returns_rrf_results() -> None:
    """Hybrid retrieval returns fused RRF results."""
    chunks = build_chunks()
    retriever = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    retriever.add_chunks(chunks)
    results = await retriever.retrieve("dense BM25 scientific retrieval", limit=3)
    assert results
    assert results[0].retriever == "rrf"


def test_rrf_promotes_shared_results() -> None:
    """RRF gives a shared chunk a positive fused score."""
    chunk = build_chunks()[0]
    fused = reciprocal_rank_fusion(
        [
            [SearchResult(chunk=chunk, score=0.9, retriever="dense")],
            [SearchResult(chunk=chunk, score=2.0, retriever="bm25")],
        ]
    )
    assert fused[0].score > 0
    assert "dense" in fused[0].path
