# SyntheticControlCueExtractor Guide

![SyntheticControlCueExtractor](../assets/synthetic-control-cue-extractor.gif)

Closes Elicit / Consensus / AJE synthetic-control gaps.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `DifferenceInDifferencesCueExtractor` and `MendelianRandomizationCueExtractor`.

## Usage

```python
from retrieval.synthetic_control_cues import SyntheticControlCueExtractor

rows = SyntheticControlCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "We built a synthetic control from the donor pool with placebo-in-space checks.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
