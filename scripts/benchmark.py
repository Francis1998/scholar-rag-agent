#!/usr/bin/env python3
"""Basic benchmark for scientific RAG throughput."""

import asyncio
import statistics
import time

from ingestion.chunking import TextChunker, stable_id
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import HyDEExpander
from retrieval.models import Document
from retrieval.sparse import BM25Retriever

ITERATIONS = 100
QUERY = "How does GraphRAG support grounded scientific retrieval?"


def build_retriever() -> HybridRetriever:
    """Build a deterministic in-memory retriever for benchmark runs."""
    document = Document(
        document_id=stable_id("benchmark-corpus", "doc"),
        title="Benchmark Corpus",
        text=(
            "GraphRAG connects scientific entities across papers. Hybrid retrieval "
            "combines dense embeddings, BM25 sparse search, HyDE query expansion, "
            "and reciprocal rank fusion for grounded literature synthesis."
        ),
        source="benchmark",
    )
    chunks = TextChunker(chunk_size=240, overlap=0).chunk(document)
    retriever = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    retriever.add_chunks(chunks)
    return retriever


async def benchmark_run() -> dict[str, float | int]:
    """Run a deterministic retrieval throughput benchmark."""
    retriever = build_retriever()
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        await retriever.retrieve(QUERY, limit=3)
        times.append(time.perf_counter() - start)

    return {
        "iterations": ITERATIONS,
        "mean_ms": round(statistics.mean(times) * 1000, 2),
        "p50_ms": round(statistics.median(times) * 1000, 2),
        "p95_ms": round(sorted(times)[int(ITERATIONS * 0.95)] * 1000, 2),
        "p99_ms": round(sorted(times)[int(ITERATIONS * 0.99)] * 1000, 2),
    }


def main() -> None:
    """Run the benchmark entrypoint."""
    results = asyncio.run(benchmark_run())
    for metric_name, metric_value in results.items():
        print(f"{metric_name:<15}: {metric_value}")


if __name__ == "__main__":
    main()
