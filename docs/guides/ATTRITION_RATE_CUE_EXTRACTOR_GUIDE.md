# AttritionRateCueExtractor Guide

![AttritionRateCueExtractor](../assets/attrition-rate-cue-extractor.gif)

CONSORT-style attrition/dropout cue extraction. Never calls the network. Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.attrition_rate_cues import AttritionRateCueExtractor

rows = AttritionRateCueExtractor().extract([{"paper_id": "p1", "abstract": "..."}])
print(rows[0].flagged)
```
