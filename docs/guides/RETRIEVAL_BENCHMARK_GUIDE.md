# Offline Retrieval Benchmarks

![Measured synthetic retrieval benchmark and quality gates](../assets/retrieval-benchmark.gif)

Compare real BM25 and default hybrid rankings on the **same labeled passages**,
inspect each miss, and fail a local CI gate when either selected retriever falls
below your thresholds. No API server, model credentials, database, network
downloads, or learned embedding model is needed.

The animation renders actual synthetic-demo output; it is not a UI recording,
published benchmark, or claim of scientific accuracy.

## Why this feature

[BEIR](https://github.com/beir-cellar/beir) provides common retrieval evaluation
across architectures and labeled datasets.
[PaperQA](https://github.com/Future-House/paper-qa) demonstrates scientific
document retrieval as part of an evidence-based answering workflow.
Their useful lesson here is to measure retrieval against explicit judgments,
rather than infer quality from a plausible answer or one successful search.

Scholar already had a synchronous Python `EvaluationHarness` and a one-document
smoke command. Neither provided a dataset-driven, installed command with
comparable real retrievers, per-case JSON artifacts, and process exit codes for
quality gates. This feature connects the existing retrievers and harness.
It is **not** a BEIR-format importer, a reproduction of either project's
published results, or a RAGAS/LLM-judge integration.

## Quick start

From the repository root, after installing Python 3.11+ and `uv`:

```bash
uv sync --locked --extra dev --python 3.12
uv run scholar-rag-eval --demo --k 1
```

The packaged synthetic tutorial has six chunks and four queries, including a
query with two relevant chunks. The version-one JSON report includes a dataset
fingerprint, configuration, per-case ranked chunk IDs, hit@k, recall@k, aggregate
scores, thresholds, and pass/fail status for each retriever.

```bash
# Real threshold gate on the toy fixture; expected to fail with exit code 1.
uv run scholar-rag-eval --demo --k 1 --min-hit-rate 1

# An explicitly less restrictive cutoff; expected to pass on this fixture.
uv run scholar-rag-eval --demo --k 2 --min-hit-rate 1 --min-recall 0.875

# Select just one baseline.
uv run scholar-rag-eval --demo --retriever bm25 --k 1
```

The first gate intentionally catches a hybrid miss. Do not change your labels
or choose thresholds just to make a real failing benchmark pass.
Running `scholar-rag-eval` with **no arguments** retains the original tab-separated
one-document smoke output. Benchmark options require `--demo` or `--dataset`;
mistyped or unsupported options fail instead of silently running the smoke demo.

## Bring your own labeled passages

Create `my-retrieval.json` with this complete, minimal schema:

```json
{
  "schema_version": 1,
  "name": "Synthetic two-passage example",
  "chunks": [
    {
      "chunk_id": "hybrid",
      "document_id": "retrieval-note",
      "title": "Synthetic hybrid note",
      "text": "Hybrid retrieval combines dense and sparse rankings.",
      "source": "synthetic:example"
    },
    {
      "chunk_id": "battery",
      "document_id": "battery-note",
      "title": "Synthetic battery note",
      "text": "Lithium ion batteries store electrochemical energy.",
      "source": "synthetic:example"
    }
  ],
  "cases": [
    {
      "case_id": "fusion",
      "query": "hybrid dense sparse retrieval",
      "relevant_chunk_ids": ["hybrid"]
    }
  ]
}
```

```bash
uv run scholar-rag-eval --dataset my-retrieval.json --retriever both --k 1 \
  --min-hit-rate 1 --min-recall 1 --output new-report.json
```

Judgments are **binary and chunk-level**, not document-level. Multiple chunks
can share `document_id`, but `chunk_id` and `case_id` must be unique. Every
relevance ID must name a supplied chunk; duplicates and missing labels are
errors. Identical queries with different case IDs/labels remain separate cases.
The command does not ingest or rechunk documents, infer labels, or fetch `source`
URLs. Export or create passages you have permission to evaluate and label them
independently. Split development and held-out judgments for a meaningful gate.

## Options, artifacts, and errors

| Option | Contract |
| --- | --- |
| `--dataset PATH` / `--demo` | Mutually exclusive explicit data sources |
| `--retriever bm25\|hybrid\|both` | Default `both`, in BM25 then hybrid order |
| `--k INTEGER` | Default 5; strict range 1-100 |
| `--min-hit-rate NUMBER` | Optional finite minimum mean hit@k, between 0 and 1 |
| `--min-recall NUMBER` | Optional finite minimum mean recall@k, between 0 and 1 |
| `--output PATH` | Also save the complete JSON to a **new** UTF-8 file |

Thresholds are inclusive and compare unrounded means. **Every** selected
retriever must satisfy **every** requested threshold. The report is always
written to stdout on a valid evaluation, including threshold failures.
`--output` preserves the same report on failure; it never overwrites the
dataset, an existing report, or an existing symlink. Its parent directory must
already exist. Diagnostic messages go to stderr, not into JSON.

| Exit code | Meaning |
| --- | --- |
| 0 | Evaluation completed and all requested thresholds passed |
| 1 | A requested quality threshold failed; inspect the complete JSON report |
| 2 | Invalid arguments/dataset or an output-file error; no successful report |

Without thresholds, `passed: true` means the evaluation completed, **not** that
retrieval quality is acceptable. Unexpected internal retrieval errors propagate
as process failures, not empty or fabricated success reports.

The loader accepts at most 8 MiB of UTF-8 JSON. The schema bounds input to
10,000 chunks, 1,000 cases, 32,768 characters per passage, 4,096 per query,
128 per ID, 512 per title, and 1,024 per source. Names are at most 128 characters.
IDs must be nonblank/unpadded; other supplied strings must be nonblank.
Unknown fields, incorrect types (including boolean schema versions), duplicate
JSON keys, non-finite JSON numbers, and invalid labels are rejected.
Metadata is intentionally outside this small interchange schema.

`dataset_sha256` hashes canonical validated JSON, not the original file's
whitespace. It includes the name, text, queries, labels, and list ordering.
Reordering chunks can change tied rankings and therefore changes the fingerprint.
This digest is not a signature or an accuracy claim. Record your code revision
(`git rev-parse HEAD`), lockfile, report, dataset, and labeling process together
for a portfolio or reproducible review.

## What is measured

BM25 uses its existing default parameters and unexpanded queries. Hybrid uses
the existing 64-dimensional lexical hash vectors, BM25, deterministic HyDE
template expansion, and reciprocal rank fusion. These are **not learned
semantic embeddings**. Fresh indexes and the same supplied corpus ordering
are used for each configuration. The entire corpus remains eligible; a cutoff
larger than the corpus naturally returns fewer than `k` results.

Hit@k records whether any labeled relevant chunk was retrieved; recall@k records
the fraction of that case's relevant chunk IDs retrieved. Aggregate scores are
unweighted means across cases. Identical queries may share a cached ranking
within one configuration, but each case keeps its own judgments and score.

This isolates candidate retrieval. It does **not** benchmark `/query` planning,
graph expansion, reranking, generation, latency, answer quality, or semantic
faithfulness. For separately supplied answer/reference lexical metrics, use the
[Python harness](EVALUATION_HARNESS_GUIDE.md). For the actual application stack
and current, officially sourced GPT/Claude/Gemini/Kimi model choices, see
[Architecture](../../ARCHITECTURE.md) and [provider models](PROVIDER_MODELS_GUIDE.md).
No provider is loaded by this command even when keys or invalid application
settings are present in the environment.

## Python API

```python
import asyncio

from evaluation.benchmark import BenchmarkOptions, load_demo_dataset, run_benchmark

report = asyncio.run(
    run_benchmark(
        load_demo_dataset(),
        BenchmarkOptions(k=2, retrievers=("bm25", "hybrid"), min_hit_at_k=1.0),
    )
)
print(report.model_dump_json(indent=2))
assert report.passed
```

Use `load_dataset(Path(...))` for local files. `run_benchmark` snapshots and
revalidates the supplied dataset/options before awaiting retrieval, so concurrent
mutation of caller-owned lists cannot change an in-flight run. Python input
models enforce the same per-record/count bounds; the 8 MiB limit applies to the
JSON file/bytes loader.

## Reproduce the animation and review artifacts

```bash
BENCHMARK_DIR="$(mktemp -d)/retrieval-benchmark"
uv run python -m scripts.demo_retrieval_benchmark --output-dir "$BENCHMARK_DIR"
uv run python -m scripts.create_retrieval_benchmark_gif \
  --transcript "$BENCHMARK_DIR/transcript.txt" \
  --output "$BENCHMARK_DIR/retrieval-benchmark.gif"
```

Inspect `dataset.json`, `baseline.json`, `gate-fail.json`, `gate-pass.json`, and
`transcript.txt`. The generator uses the shared Pillow renderer and refuses
clipped text, unrelated transcripts, or existing output files. Its labels
explicitly distinguish an illustration from a screen recording.

For a portfolio, show one meaningful miss and its source passages, justify the
judgments and thresholds, and compare a proposed retrieval change against the
same dataset. A tiny synthetic corpus demonstrates engineering, not scientific
performance. Reports omit passage/query text, but dataset names, IDs, and hashes
can still identify private research; review artifacts before sharing.
