# AttritionInformativeCensoringCueExtractor Guide

![AttritionInformativeCensoringCueExtractor demo](../assets/attrition-informative-censoring-cues.gif)

Offline deterministic cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE attrition informative-censoring cue extractors.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.attrition_informative_censoring_cues import AttritionInformativeCensoringCueExtractor

cues = AttritionInformativeCensoringCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Bias may arise from informative censoring in the cohort."}]
)
print(cues[0].flagged, cues[0].cue_kind)
```
