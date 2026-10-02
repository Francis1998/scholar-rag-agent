# RegressionDiscontinuityCueExtractor Guide

![RegressionDiscontinuityCueExtractor](../assets/regression-discontinuity-cue-extractor.gif)

Closes Elicit / Consensus / AJE regression-discontinuity gaps.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `DifferenceInDifferencesCueExtractor` and `SyntheticControlCueExtractor`.

## Usage

```python
from retrieval.regression_discontinuity_cues import RegressionDiscontinuityCueExtractor

rows = RegressionDiscontinuityCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "We used a regression discontinuity design with the running variable and Imbens-Kalyanaraman bandwidth.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
