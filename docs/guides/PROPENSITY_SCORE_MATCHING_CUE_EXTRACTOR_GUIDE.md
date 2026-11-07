# PropensityScoreMatchingCueExtractor Guide

![PropensityScoreMatchingCueExtractor](../assets/propensity-score-matching-cue-extractor.gif)

Closes Elicit / Consensus / AJE propensity-score matching gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ConfoundingAdjustmentCueExtractor` and `NegativeControlOutcomeCueExtractor`.

## Usage

```python
from retrieval.propensity_score_matching_cues import PropensityScoreMatchingCueExtractor

rows = PropensityScoreMatchingCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "We applied propensity score matching with a caliper and checked standardized mean difference balance.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
