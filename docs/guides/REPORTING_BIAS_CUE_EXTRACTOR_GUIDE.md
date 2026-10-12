# ReportingBiasCueExtractor Guide

![ReportingBiasCueExtractor demo](../assets/reporting-bias-cues.gif)

Offline deterministic cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE reporting-bias cue extractors.

Optional later narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.reporting_bias_cues import ReportingBiasCueExtractor

cues = ReportingBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Results may reflect reporting bias in selective endpoints."}]
)
print(cues[0].flagged, cues[0].cue_kind)
```
