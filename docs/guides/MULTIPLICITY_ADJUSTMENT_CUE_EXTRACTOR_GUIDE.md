# MultiplicityAdjustmentCueExtractor Guide

![MultiplicityAdjustmentCueExtractor](../assets/multiplicity-adjustment-cue-extractor.gif)

Offline evidence cue extractor. Never calls the network.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from PValueHintExtractor and SubgroupAnalysisCueExtractor.

## Usage

```python
from retrieval.multiplicity_adjustment_cues import MultiplicityAdjustmentCueExtractor

rows = MultiplicityAdjustmentCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "P-values used Bonferroni correction."}]
)
assert rows[0].flagged is True
```
