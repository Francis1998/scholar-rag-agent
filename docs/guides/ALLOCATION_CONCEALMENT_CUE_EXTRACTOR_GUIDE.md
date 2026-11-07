# AllocationConcealmentCueExtractor Guide

![AllocationConcealmentCueExtractor](../assets/allocation-concealment-cue-extractor.gif)

Offline evidence cue extractor. Never calls the network.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from BlindingStatusCueExtractor and RiskOfBiasCueExtractor.

## Usage

```python
from retrieval.allocation_concealment_cues import AllocationConcealmentCueExtractor

rows = AllocationConcealmentCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Allocation concealment used sealed opaque envelopes."}]
)
assert rows[0].flagged is True
```
