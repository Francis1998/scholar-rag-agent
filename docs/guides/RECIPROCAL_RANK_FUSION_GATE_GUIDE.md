# ReciprocalRankFusionGate Guide

![ReciprocalRankFusionGate demo](../assets/reciprocal-rank-fusion-gate.gif)

Provider-independent local retrieval postprocessor, not a DOI connector.
See the [provider guide](PROVIDER_MODELS_GUIDE.md) for supported model adapters.

## Ranking behavior

The gate uses the shared reciprocal-rank-fusion helper. Each exact chunk ID
contributes one vote per input ranked list, at its first (best) original
1-based rank. Repeating an ID within a list does not add votes or renumber
later hits. Identical text with different chunk IDs remains distinct.

A chunk found in two lists still receives two votes, even when both lists
use the same retriever label. The output `path` contains the first occurrence's
label from each contributing list, in input-list order, and retains the first
occurrence's chunk metadata. Results use `retriever="reciprocal_rank_fusion_gate"`;
score ties preserve first-seen order. Inputs are not mutated.

The `top_k` cap applies after fusion. `None` (the default) returns all results;
zero or negative values return an empty list.

## Usage

```python
from retrieval.reciprocal_rank_fusion_gate import ReciprocalRankFusionGate

gate = ReciprocalRankFusionGate(k=60)
fused = gate.gate([dense_results, bm25_results], top_k=10)
```

See unit tests for edge cases.
