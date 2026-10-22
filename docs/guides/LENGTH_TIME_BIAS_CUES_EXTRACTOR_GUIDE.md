# LengthTimeBiasCueExtractor Guide

![LengthTimeBiasCueExtractor](../../assets/demo/length-time-bias-cues.gif)

Offline HITL helper for `cues` bands.
Gap vs Elicit/Consensus/Connected Papers length-time bias cue extractors. Distinct from `LeadTimeBiasCueExtractor` / `DetectionBiasCueExtractor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.length_time_bias_cues import LengthTimeBiasCueExtractor

rows = LengthTimeBiasCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Authors note length time bias concerns.",
        }
    ],
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
