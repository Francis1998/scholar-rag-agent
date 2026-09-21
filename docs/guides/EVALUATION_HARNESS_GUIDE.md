# Evaluation Harness Guide

![Evaluation harness demo](../assets/evaluation-harness.gif)

`EvaluationHarness` runs offline hit-rate@k, recall@k, generated-answer
faithfulness, and separate reference-answer coverage over labeled cases.
Inspired by RAGAS / PaperQA evaluation loops and BEIR-style retrieval metrics.
It is provider-independent and makes no network or LLM calls; generated
answers must be supplied by the caller. See the
[provider models guide](PROVIDER_MODELS_GUIDE.md) for provider configuration.

## Usage

```python
from evaluation.harness import EvalCase, EvaluationHarness
from retrieval.models import Chunk, SearchResult


def retrieve(_query: str, k: int) -> list[SearchResult]:
    """Synchronous offline fixture; replace with your retrieval callback."""
    return [
        SearchResult(
            chunk=Chunk(
                chunk_id="chunk-hybrid",
                document_id="paper-1",
                title="Hybrid retrieval",
                text="Hybrid retrieval combines dense and sparse evidence.",
                source="fixture",
            ),
            score=1.0,
            retriever="fixture",
        )
    ][:k]


harness = EvaluationHarness(
    [
        EvalCase(
            case_id="q1",
            query="What fuses dense and sparse evidence?",
            relevant_chunk_ids=frozenset({"chunk-hybrid"}),
            gold_answer="hybrid retrieval",
        )
    ]
)
report = harness.evaluate(retrieve, k=5, answers={"q1": "hybrid astronomy"})
print(report.mean_hit_at_k, report.mean_recall_at_k)  # 1.0 1.0
print(report.mean_faithfulness)  # 0.5: only "hybrid" is in the evidence
print(report.mean_reference_coverage)  # 1.0: both gold-answer tokens are present
```

Run the example in an environment with the project installed
(`uv sync --locked --extra dev --python 3.12`). The callback must be synchronous:
the harness does not await `HybridRetriever.retrieve` or other asynchronous
retrievers. An async application can await retrieval first, cache the resulting
lists by query, and pass a synchronous lookup callback to the harness.

## Metric contract

The harness uses the first `k` results in callback order for **all** metrics.
It does not rerank them. Hit-rate@k and recall@k still compare retrieved chunk
IDs with `relevant_chunk_ids`.

| Metric | Text scored against retrieved titles and text | Missing input | Supplied empty or tokenless input |
| --- | --- | --- | --- |
| `CaseScore.faithfulness` | Generated answer from `answers[case_id]` | `None` | `0.0` |
| `CaseScore.reference_coverage` | Reference answer from `EvalCase.gold_answer` | `None` | `0.0` |

Each text score is the fraction of **unique content tokens** in its own input
that appear anywhere in the retrieved titles and text. The
[implementation](../../src/evaluation/harness.py) lowercases text, extracts
ASCII `[a-z0-9]+` tokens, and removes its fixed English stopword set. Repeated
tokens count once. Nonempty content-token input with no evidence scores `0.0`.

`gold_answer` never substitutes for the generated answer or changes its
faithfulness. Reference coverage is computed even when no generated answer is
provided. `answers=None`, an empty answer map, or a missing case key leave that
case's faithfulness unset; an explicitly supplied empty answer scores zero.
`mean_faithfulness` and `mean_reference_coverage` independently average only
non-`None` scores, including zeros, and are `None` when no cases are scored.

**Limitations:** as the source's set-overlap implementation shows, these are
lexical coverage proxies, **not truth, correctness, or entailment checks**.
They do not resolve contradictions, negation, synonyms, or claim-to-citation
alignment. A high reference-coverage score says nothing about the quality of
the generated answer, and even full lexical faithfulness is not proof that
an answer is supported semantically.

## Compatibility

The optional third `gold_answer` argument to
`EvaluationHarness.faithfulness(answer, results, gold_answer)` remains accepted,
positionally or by keyword, but is now ignored. Callers that relied on gold
token coverage should read `reference_coverage` / `mean_reference_coverage`
from `evaluate` instead.

The new fields are appended to `CaseScore` and `HarnessReport` with `None`
defaults, preserving existing positional and keyword construction. Serialized
dataclasses now include these additional fields; strict consumers must allow
them. Existing generated-answer scores without gold, retrieval metrics, ranking,
and cutoff behavior are unchanged.
