# NegativeControlOutcomeCueExtractor Guide

![NegativeControlOutcomeCueExtractor](../assets/negative-control-outcome-cue-extractor.gif)

Closes Elicit / Consensus / AJE negative-control outcome gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `TransportabilityCueExtractor` and `ConfoundingAdjustmentCueExtractor`.

## Usage

```python
from retrieval.negative_control_outcome_cues import NegativeControlOutcomeCueExtractor

rows = NegativeControlOutcomeCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "We used a negative control outcome and falsification test for bias.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
