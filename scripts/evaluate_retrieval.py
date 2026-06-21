"""Evaluate retrieval quality on a small deterministic fixture corpus."""

import asyncio

from ingestion.chunking import TextChunker, stable_id
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Document
from retrieval.sparse import BM25Retriever


async def evaluate() -> None:
    """Run a smoke retrieval evaluation and print ranked chunk ids."""
    document = Document(
        document_id=stable_id("evaluation", "doc"),
        title="Evaluation Fixture",
        text="Hybrid retrieval uses dense and sparse evidence for grounded scientific question answering.",
        source="fixture",
    )
    chunks = TextChunker(chunk_size=200, overlap=0).chunk(document)
    retriever = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    retriever.add_chunks(chunks)
    results = await retriever.retrieve("What combines dense and sparse evidence?", limit=3)
    for result in results:
        print(f"{result.chunk.chunk_id}\t{result.score:.4f}\t{result.retriever}")


def main() -> None:
    """Run retrieval evaluation."""
    asyncio.run(evaluate())


if __name__ == "__main__":
    main()
