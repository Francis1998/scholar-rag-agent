#!/usr/bin/env python3
"""Basic benchmark for scientific RAG throughput."""

import statistics
import time

ITERATIONS = 100


def benchmark_run() -> dict[str, float | int]:
    """Run a basic throughput benchmark."""
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        # Placeholder: replace with actual operation
        time.sleep(0.001)
        times.append(time.perf_counter() - start)

    return {
        "iterations": ITERATIONS,
        "mean_ms": round(statistics.mean(times) * 1000, 2),
        "p50_ms": round(statistics.median(times) * 1000, 2),
        "p95_ms": round(sorted(times)[int(ITERATIONS * 0.95)] * 1000, 2),
        "p99_ms": round(sorted(times)[int(ITERATIONS * 0.99)] * 1000, 2),
    }


if __name__ == "__main__":
    results = benchmark_run()
    for metric_name, metric_value in results.items():
        print(f"{metric_name:<15}: {metric_value}")
