# IntentionToTreatCueExtractor Guide

![IntentionToTreatCueExtractor demo](../assets/intention-to-treat-cue-extractor.gif)

`IntentionToTreatCueExtractor` Extract offline ITT/mITT/per-protocol analysis cues; fills an Elicit/Consensus/Cochrane/PaperQA ITT-analysis gap (distinct from BlindingStatusCueExtractor / RiskOfBiasCueExtractor)

Never calls the network. Optional later narrative: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.intention_to_treat_cues import IntentionToTreatCueExtractor

rows = IntentionToTreatCueExtractor().extract([{"paper_id": "p1", "abstract": "..."}])
print(rows[0].flagged, rows[0].cues)
```
