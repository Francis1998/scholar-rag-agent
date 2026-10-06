# ColliderStratificationCueExtractor Guide

![ColliderStratificationCueExtractor flow](../assets/collider-stratification-cue-extractor.gif)

Offline cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE collider-stratification cue extractors.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `EValueSensitivityCueExtractor / SpilloverInterferenceCueExtractor`.

## Usage

```python
from retrieval.collider_stratification_cues import ColliderStratificationCueExtractor

cues = ColliderStratificationCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "We discuss collider stratification and Berkson's bias.",
        }
    ]
)
assert cues[0].flagged is True
```
