# Evaluation Harness Guide

![Evaluation harness demo](../assets/evaluation-harness.gif)

`EvaluationHarness` runs offline hit-rate@k, recall@k, and lexical answer
faithfulness over labeled cases. Inspired by RAGAS / PaperQA evaluation loops
and BEIR-style retrieval metrics. Distinct from the smoke script
`scripts/evaluate_retrieval.py`. Local harness for GPT-5.5 / Claude Sonnet 4.6 /
Gemini 3.x / Kimi K2 scholarly RAG evaluation (not a DOI connector).

## Usage

```python
from evaluation.harness import EvalCase, EvaluationHarness

harness = EvaluationHarness([
    EvalCase(
        case_id="q1",
        query="What fuses dense and sparse evidence?",
        relevant_chunk_ids=frozenset({"chunk-hybrid"}),
        gold_answer="hybrid retrieval",
    )
])
report = harness.evaluate(retriever.retrieve, k=5, answers={"q1": "hybrid retrieval"})
print(report.mean_hit_at_k, report.mean_recall_at_k, report.mean_faithfulness)
```

Faithfulness is the fraction of gold-answer content tokens found in retrieved
chunk text. No network or LLM call is required.
