# AdverseEventCueExtractor Guide

![AdverseEventCueExtractor](../assets/adverse-event-cue-extractor.gif)

Offline adverse-event reporting cue extraction. Never network. Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.adverse_event_cues import AdverseEventCueExtractor

print(AdverseEventCueExtractor().extract([{"paper_id": "p1", "abstract": "..."}]))
```
