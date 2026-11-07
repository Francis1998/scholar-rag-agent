# OverdiagnosisBiasCueExtractor Guide

![OverdiagnosisBiasCueExtractor demo](../assets/overdiagnosis-bias-cues.gif)

Offline deterministic cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE overdiagnosis-bias cue extractors.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.overdiagnosis_bias_cues import OverdiagnosisBiasCueExtractor

cues = OverdiagnosisBiasCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Outcome rates may reflect overdiagnosis bias in screening cohorts.",
        }
    ]
)
print(cues[0].flagged, cues[0].cue_kind)
```
