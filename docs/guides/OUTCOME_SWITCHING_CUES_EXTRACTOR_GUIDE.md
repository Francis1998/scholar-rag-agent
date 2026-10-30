# OutcomeSwitchingCueExtractor Guide

![OutcomeSwitchingCueExtractor](../../assets/demo/outcome-switching-cues.gif)

Offline cue extractor.
Gap vs Elicit/Consensus/Connected Papers outcome-switching cue extractors. Distinct from `PrimaryEndpointCueExtractor` / `SpinPHackingCueExtractor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.outcome_switching_cues import OutcomeSwitchingCueExtractor

rows = OutcomeSwitchingCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss outcome switching."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O.
