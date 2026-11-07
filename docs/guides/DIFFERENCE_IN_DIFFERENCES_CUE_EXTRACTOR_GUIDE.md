# DifferenceInDifferencesCueExtractor Guide

![DifferenceInDifferencesCueExtractor](../assets/difference-in-differences-cue-extractor.gif)

Closes Elicit / Consensus / AJE difference-in-differences gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `TimeVaryingConfoundingCueExtractor` and `ClusterRandomizationCueExtractor`.

## Usage

```python
from retrieval.difference_in_differences_cues import DifferenceInDifferencesCueExtractor

rows = DifferenceInDifferencesCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "We estimated difference-in-differences under parallel trends with TWFE.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
