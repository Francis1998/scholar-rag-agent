# FundingConflictCueExtractor Guide

![FundingConflictCueExtractor](../assets/funding-conflict-cue-extractor.gif)

Offline evidence cue extractor. Never calls the network.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from OpenDataAvailabilityFlagger and RetractionFlagger.

## Usage

```python
from retrieval.funding_conflict_cues import FundingConflictCueExtractor

rows = FundingConflictCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Industry-funded trial with COI disclosure."}]
)
assert rows[0].flagged is True
```
