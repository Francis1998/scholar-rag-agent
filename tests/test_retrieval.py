"""Tests for hybrid retrieval components."""

import pytest

from ingestion.chunking import TextChunker, stable_id
from retrieval.dense import DenseRetriever
from retrieval.embeddings import HashEmbeddingModel, cosine_similarity
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Document, SearchResult
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.sparse import BM25Retriever


def test_hash_embedding_is_invariant_to_attached_punctuation() -> None:
    """Trailing punctuation must not change a token's embedding dimension.

    The dense embedder previously split on raw whitespace, so ``retrieval.``
    hashed to a different dimension than ``retrieval`` and the same word was a
    hit in BM25 sparse retrieval but a miss in the dense vector the two are
    fused with. Tokenizing with the shared sparse tokenizer makes the embedding
    of a phrase identical whether or not its terms carry attached punctuation.
    """
    embedder = HashEmbeddingModel()

    plain = embedder.embed("machine learning")
    punctuated = embedder.embed("machine, learning.")

    assert cosine_similarity(plain, punctuated) == pytest.approx(1.0)


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
