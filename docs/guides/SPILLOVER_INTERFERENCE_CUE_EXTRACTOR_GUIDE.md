# SpilloverInterferenceCueExtractor Guide

![SpilloverInterferenceCueExtractor flow](../assets/spillover-interference-cue-extractor.gif)

Offline cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE spillover / interference / SUTVA cue extractors.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `SyntheticControlCueExtractor / IntentionToTreatCueExtractor`.

## Usage

```python
from retrieval.spillover_interference_cues import SpilloverInterferenceCueExtractor

cues = SpilloverInterferenceCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "We observed a treatment spillover and interference between units violating SUTVA.",
        }
    ]
)
assert cues[0].flagged is True
```
