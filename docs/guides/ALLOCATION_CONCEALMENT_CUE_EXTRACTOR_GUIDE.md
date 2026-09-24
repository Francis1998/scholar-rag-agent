# AllocationConcealmentCueExtractor Guide

![AllocationConcealmentCueExtractor](../assets/allocation-concealment-cue-extractor.gif)

Offline evidence cue extractor. Never calls the network.

Optional later narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from BlindingStatusCueExtractor and RiskOfBiasCueExtractor.

## Usage

```python
from retrieval.allocation_concealment_cues import AllocationConcealmentCueExtractor

rows = AllocationConcealmentCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Allocation concealment used sealed opaque envelopes."}]
)
assert rows[0].flagged is True
```
