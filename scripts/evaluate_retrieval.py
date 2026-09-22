"""Run a legacy smoke demo or a labeled, model-free retrieval benchmark."""

import argparse
import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

from evaluation.benchmark import (
    BenchmarkOptions,
    RetrieverName,
    load_dataset,
    load_demo_dataset,
    run_benchmark,
)
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
        text=(
            "Hybrid retrieval uses dense and sparse evidence for grounded scientific "
            "question answering."
        ),
        source="fixture",
    )
    chunks = TextChunker(chunk_size=200, overlap=0).chunk(document)
    retriever = HybridRetriever(DenseRetriever(), BM25Retriever(), HyDEExpander())
    retriever.add_chunks(chunks)
    results = await retriever.retrieve("What combines dense and sparse evidence?", limit=3)
    for result in results:
        print(f"{result.chunk.chunk_id}\t{result.score:.4f}\t{result.retriever}")


def main(argv: list[str] | None = None) -> None:
    """Preserve no-argument smoke output; benchmark explicit datasets with CI exit codes."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path, help="Version-one labeled UTF-8 JSON dataset")
    source.add_argument("--demo", action="store_true", help="Use the packaged synthetic dataset")
    parser.add_argument("--k", type=int, help="Retrieval cutoff, from 1 through 100 (default: 5)")
    parser.add_argument("--retriever", choices=["bm25", "hybrid", "both"], help="Default: both")
    parser.add_argument("--min-hit-rate", type=float, help="Minimum mean hit@k for every retriever")
    parser.add_argument(
        "--min-recall", type=float, help="Minimum mean recall@k for every retriever"
    )
    parser.add_argument("--output", type=Path, help="Also save JSON to a NEW file; never overwrite")
    arguments = parser.parse_args(argv)
    if arguments.dataset is None and not arguments.demo:
        if any(
            value is not None
            for value in (
                arguments.k,
                arguments.retriever,
                arguments.min_hit_rate,
                arguments.min_recall,
                arguments.output,
            )
        ):
            parser.error("benchmark options require --dataset or --demo")
        asyncio.run(evaluate())
        return
    selections: dict[str, tuple[RetrieverName, ...]] = {
        "bm25": ("bm25",),
        "hybrid": ("hybrid",),
        "both": ("bm25", "hybrid"),
    }
    try:
        options = BenchmarkOptions(
            k=5 if arguments.k is None else arguments.k,
            retrievers=selections[arguments.retriever or "both"],
            min_hit_at_k=arguments.min_hit_rate,
            min_recall_at_k=arguments.min_recall,
        )
    except ValidationError as exc:
        parser.error(str(exc))
    try:
        dataset = load_demo_dataset() if arguments.demo else load_dataset(arguments.dataset)
    except (OSError, ValueError) as exc:
        parser.error(f"invalid dataset: {exc}")
    report = asyncio.run(run_benchmark(dataset, options))
    output = report.model_dump_json(indent=2) + "\n"
    if arguments.output is not None:
        try:
            with arguments.output.open("x", encoding="utf-8") as destination:
                destination.write(output)
        except OSError as exc:
            parser.error(f"cannot create report: {exc}")
    sys.stdout.write(output)
    if not report.passed:
        print("Retrieval benchmark failed the requested quality thresholds.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
