# AscertainmentBiasCueExtractor Guide

![AscertainmentBiasCueExtractor demo](../assets/ascertainment-bias-cues.gif)

Offline deterministic cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE ascertainment-bias cue extractors.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.ascertainment_bias_cues import AscertainmentBiasCueExtractor

cues = AscertainmentBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Outcome rates may reflect ascertainment bias in claims data."}]
)
print(cues[0].flagged, cues[0].cue_kind)
```
