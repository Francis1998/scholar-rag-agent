# NegativeControlExposureCueExtractor Guide

![NegativeControlExposureCueExtractor flow](../assets/negative-control-exposure-cue-extractor.gif)

Offline cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE negative-control exposure cue extractors.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `NegativeControlOutcomeCueExtractor / ConfoundingAdjustmentCueExtractor`.

## Usage

```python
from retrieval.negative_control_exposure_cues import NegativeControlExposureCueExtractor

cues = NegativeControlExposureCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We included a negative control exposure check."}]
)
assert cues[0].flagged is True
```

## Safety

Advisory cues only. No HTTP. Humans decide. See `SAFETY.md`.
