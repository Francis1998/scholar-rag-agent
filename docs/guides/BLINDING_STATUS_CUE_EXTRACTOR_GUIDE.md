# BlindingStatusCueExtractor Guide

![BlindingStatusCueExtractor demo](../assets/blinding-status-cue-extractor.gif)

`BlindingStatusCueExtractor` Extract offline single/double/triple-blind and open-label cues; fills an Elicit/Consensus/Cochrane/PaperQA blinding-status gap (distinct from RiskOfBiasCueExtractor)

Never calls the network. Optional later narrative: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.blinding_status_cues import BlindingStatusCueExtractor

rows = BlindingStatusCueExtractor().extract([{"paper_id": "p1", "abstract": "..."}])
print(rows[0].flagged, rows[0].cues)
```
