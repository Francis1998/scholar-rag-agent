# EstimandIchE9CueExtractor Guide

![EstimandIchE9CueExtractor](../assets/estimand-ich-e9-cue-extractor.gif)

Offline cue extractor. Never network. Closes Elicit / Consensus / ICH-E9(R1) estimand gaps.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `IntentionToTreatCueExtractor / PrimaryEndpointCueExtractor`.

## Usage

```python
from retrieval.estimand_ich_e9_cues import EstimandIchE9CueExtractor

rows = EstimandIchE9CueExtractor().extract(
    [{"paper_id": "p1", "methods": "The treatment-policy estimand was pre-specified."}]
)
print(rows[0].flagged, rows[0].cues)
```
