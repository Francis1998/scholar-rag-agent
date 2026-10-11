# AscertainmentBiasCueExtractor Guide

![AscertainmentBiasCueExtractor demo](../assets/ascertainment-bias-cues.gif)

Offline deterministic cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE ascertainment-bias cue extractors.

Optional later narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.ascertainment_bias_cues import AscertainmentBiasCueExtractor

cues = AscertainmentBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Outcome rates may reflect ascertainment bias in claims data."}]
)
print(cues[0].flagged, cues[0].cue_kind)
```
