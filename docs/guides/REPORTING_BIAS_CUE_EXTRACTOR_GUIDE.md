# ReportingBiasCueExtractor Guide

![ReportingBiasCueExtractor demo](../assets/reporting-bias-cues.gif)

Offline deterministic cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE reporting-bias cue extractors.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.reporting_bias_cues import ReportingBiasCueExtractor

cues = ReportingBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Results may reflect reporting bias in selective endpoints."}]
)
print(cues[0].flagged, cues[0].cue_kind)
```
