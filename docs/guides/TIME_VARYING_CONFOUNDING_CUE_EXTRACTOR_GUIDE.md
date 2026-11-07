# TimeVaryingConfoundingCueExtractor Guide

![TimeVaryingConfoundingCueExtractor](../assets/time-varying-confounding-cue-extractor.gif)

Closes Elicit / Consensus / AJE time-varying confounding gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ConfoundingAdjustmentCueExtractor` and `MediationAnalysisCueExtractor`.

## Usage

```python
from retrieval.time_varying_confounding_cues import TimeVaryingConfoundingCueExtractor

rows = TimeVaryingConfoundingCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "We fit a marginal structural model for time-varying confounding.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
