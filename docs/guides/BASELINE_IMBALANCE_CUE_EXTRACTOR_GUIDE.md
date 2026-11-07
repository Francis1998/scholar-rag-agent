# BaselineImbalanceCueExtractor Guide

![BaselineImbalanceCueExtractor](../assets/baseline-imbalance-cue-extractor.gif)

Offline evidence-methods cue extractor for baseline / covariate / Table-1
imbalance. Never network I/O. Gap vs Elicit / Consensus / CONSORT baseline
reporting.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `RiskOfBiasCueExtractor` and `SubgroupAnalysisCueExtractor`.

## Usage

```python
from retrieval.baseline_imbalance_cues import BaselineImbalanceCueExtractor

rows = BaselineImbalanceCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "There was baseline imbalance in age."}]
)
assert rows[0].flagged is True
```
