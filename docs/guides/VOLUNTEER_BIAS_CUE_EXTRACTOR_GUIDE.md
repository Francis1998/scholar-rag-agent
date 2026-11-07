# VolunteerBiasCueExtractor Guide

![VolunteerBiasCueExtractor](../assets/volunteer-bias-cue-extractor.gif)

Offline deterministic cue extractor. Never network I/O.
Gap vs Elicit/Consensus/AJE volunteer-bias cue extractors. Distinct from healthy-worker-effect and selection-bias cue extractors.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.volunteer_bias_cues import VolunteerBiasCueExtractor

cues = VolunteerBiasCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Enrollment may suffer from volunteer bias among health-conscious participants.",
        }
    ]
)
assert cues[0].flagged is True
```
