# Answerability Gate Guide

![Answerability gate demo](../assets/answerability_gate.gif)

`AnswerabilityGate` checks whether retrieved chunks can lexically support
answering a query before synthesis. It is inspired by LlamaIndex answerability
checks and Self-RAG "can answer?" gates, but requires no network call or LLM:
scores are pure term overlap via `retrieval.sparse.meaningful_terms`. Accepted
evidence can then be handed to GPT-5.5, Claude Sonnet 4.6, Gemini 3.x, or
Kimi K2 for grounded synthesis.

## How scoring works

1. Extract distinct non-stopword terms from the query.
2. For each result, compute coverage against the chunk title and text:

```text
result_score = |query_terms ∩ chunk_terms| / |query_terms|
```

3. Aggregate `answerability` is the **mean** of those per-result scores
   (`0.0` when there are no results or the query has no meaningful terms).
4. `kept_count` / `dropped_count` count how many per-result scores meet
   `min_score` (independent of the aggregate threshold).

## How filtering works

`filter(query, results)` first calls `score`. Then:

- If `answerability < answerability_threshold`, return an **empty list** for
  the whole batch (even if some individual chunks would pass `min_score`).
- Otherwise return the original result objects whose per-result score is at
  least `min_score`, preserving input order and object identity.

```text
if mean(result_scores) < answerability_threshold:
    return []
else:
    return [r for r, s in zip(results, result_scores) if s >= min_score]
```

This differs from `CorrectiveRagGate`, which grades keep/filter/retry signals
without a whole-batch mean threshold, and from `SelfRagReflectionGate`, which
emits SUPPORT/PARTIAL/REFUSE with conflict detection.

## Example

```python
from retrieval.answerability_gate import AnswerabilityGate

gate = AnswerabilityGate(min_score=0.2, answerability_threshold=0.3)
report = gate.score(query, retrieved_results)
if report.answerability < 0.3:
    refuse_or_rewrite()
else:
    synthesize(gate.filter(query, retrieved_results))
```

## Configuration notes

- `min_score` and `answerability_threshold` are inclusive finite values in
  `[0.0, 1.0]`; invalid values raise `ValueError` at construction.
- Titles participate in coverage so concise scholarly records can qualify.
- Blank or stopword-only queries score `0.0` and filter to an empty list.
- Empty input returns an empty report / empty list.
