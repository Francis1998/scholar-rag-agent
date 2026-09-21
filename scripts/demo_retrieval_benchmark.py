"""Record real synthetic retrieval scores and both failing and passing quality gates."""

import argparse
import asyncio
from pathlib import Path

from evaluation.benchmark import BenchmarkOptions, BenchmarkReport, load_demo_dataset, run_benchmark

PANEL_TITLES = (
    "1. Load an explicitly labeled synthetic corpus",
    "2. Compare real local retriever rankings",
    "3. Keep a failing quality gate visible",
    "4. Inspect a passing gate and its limits",
)


async def _measure() -> tuple[BenchmarkReport, BenchmarkReport, BenchmarkReport]:
    dataset = load_demo_dataset()
    baseline = await run_benchmark(dataset, BenchmarkOptions(k=1))
    failing = await run_benchmark(dataset, BenchmarkOptions(k=1, min_hit_at_k=1.0))
    passing = await run_benchmark(
        dataset, BenchmarkOptions(k=2, min_hit_at_k=1.0, min_recall_at_k=0.875)
    )
    return baseline, failing, passing


def run_demo(output_dir: Path) -> str:
    """Save measured reports and a transcript without touching settings, models, or storage."""
    filenames = (
        "dataset.json",
        "baseline.json",
        "gate-fail.json",
        "gate-pass.json",
        "transcript.txt",
    )
    for filename in filenames:
        target = output_dir / filename
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}.")
    baseline, failing, passing = asyncio.run(_measure())
    if failing.passed or not passing.passed:
        raise RuntimeError(
            "Synthetic gate outcomes changed; inspect rankings before updating the demo."
        )
    panels = [
        "\n".join(
            [
                PANEL_TITLES[0],
                f"Chunks: {baseline.chunk_count}; labeled queries: {baseline.case_count}.",
                f"Dataset SHA256: {baseline.dataset_sha256}",
                "No models, provider credentials, network calls, or database.",
                "Synthetic tutorial labels, not an accuracy benchmark.",
            ]
        ),
        "\n".join(
            [
                PANEL_TITLES[1],
                f"Cutoff k={baseline.k}; fresh indexes use identical corpus order.",
                *(
                    f"{score.name}: hit@k={score.mean_hit_at_k:.3f}; "
                    f"recall@k={score.mean_recall_at_k:.3f}"
                    for score in baseline.retrievers
                ),
                "Full ranked IDs and per-case results: baseline.json.",
                "Hybrid uses lexical hash vectors, not learned embeddings.",
            ]
        ),
        "\n".join(
            [
                PANEL_TITLES[2],
                f"At k={failing.k}, require mean hit@k >= {failing.min_hit_at_k}.",
                *(
                    f"{score.name}: {'PASS' if score.passed else 'FAIL'}"
                    for score in failing.retrievers
                ),
                f"All selected retrievers pass: {failing.passed}.",
                "The failed gate is preserved in gate-fail.json.",
            ]
        ),
        "\n".join(
            [
                PANEL_TITLES[3],
                f"At k={passing.k}, require hit@k >= {passing.min_hit_at_k}, "
                f"recall@k >= {passing.min_recall_at_k}.",
                f"All selected retrievers pass: {passing.passed}.",
                "Configuration, thresholds, and ranked IDs: gate-pass.json.",
                "Review labels and misses; do not optimize for this toy fixture.",
                "No answer generation, semantic faithfulness, or latency claim.",
            ]
        ),
    ]
    transcript = "\n\n".join(panels) + "\n"
    payloads = {
        "dataset.json": load_demo_dataset().model_dump_json(indent=2) + "\n",
        "baseline.json": baseline.model_dump_json(indent=2) + "\n",
        "gate-fail.json": failing.model_dump_json(indent=2) + "\n",
        "gate-pass.json": passing.model_dump_json(indent=2) + "\n",
        "transcript.txt": transcript,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in payloads.items():
        with (output_dir / filename).open("x", encoding="utf-8") as output:
            output.write(content)
    return transcript


def main() -> None:
    """Write the reproducible benchmark demonstration into a fresh artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    print(run_demo(arguments.output_dir), end="")


if __name__ == "__main__":
    main()
