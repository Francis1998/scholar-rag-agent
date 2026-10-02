# InterruptedTimeSeriesCueExtractor Guide

![InterruptedTimeSeriesCueExtractor](../assets/interrupted-time-series-cue-extractor.gif)

Closes Elicit / Consensus / AJE interrupted-time-series gaps.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `DifferenceInDifferencesCueExtractor` and `SyntheticControlCueExtractor`.

## Usage

```python
from retrieval.interrupted_time_series_cues import InterruptedTimeSeriesCueExtractor

rows = InterruptedTimeSeriesCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "An interrupted time series showed a level change and slope change after intervention.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
