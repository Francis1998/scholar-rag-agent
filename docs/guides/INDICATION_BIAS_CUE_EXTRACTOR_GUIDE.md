# IndicationBiasCueExtractor Guide

![IndicationBiasCueExtractor demo](../assets/indication-bias-cues.gif)

Offline deterministic cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE indication-bias cue extractors.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.indication_bias_cues import IndicationBiasCueExtractor

cues = IndicationBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Cohorts may reflect indication bias in treated patients."}]
)
print(cues[0].flagged, cues[0].cue_kind)
```
