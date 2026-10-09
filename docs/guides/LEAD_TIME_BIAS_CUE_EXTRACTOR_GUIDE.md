# LeadTimeBiasCueExtractor Guide

![LeadTimeBiasCueExtractor offline demo](../assets/lead-time-bias-cue-extractor.gif)

Offline deterministic cue extractor. Never calls the network.
Closes gaps vs Elicit/Consensus/AJE lead-time-bias cue extractors.

Optional later narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.lead_time_bias_cues import LeadTimeBiasCueExtractor

cues = LeadTimeBiasCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Survival gains may reflect lead-time bias from earlier detection.",
        }
    ]
)
print(cues[0].flagged, cues[0].cue_kind)
```

## Safety

Advisory cues only. Humans decide relevance. See `SAFETY.md`.
