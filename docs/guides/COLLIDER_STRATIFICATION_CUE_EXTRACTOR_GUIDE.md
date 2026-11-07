# ColliderStratificationCueExtractor Guide

![ColliderStratificationCueExtractor flow](../assets/collider-stratification-cue-extractor.gif)

Offline cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE collider-stratification cue extractors.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

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
