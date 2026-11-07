# LeftTruncationCueExtractor Guide

![LeftTruncationCueExtractor flow](../assets/left-truncation-cue-extractor.gif)

Offline cue extractor. Never network I/O.

Gap vs Elicit/Consensus/AJE left-truncation / delayed-entry cue extractors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.left_truncation_cues import LeftTruncationCueExtractor

cues = LeftTruncationCueExtractor().extract([{"paper_id": "p1", "abstract": "..."}])
print(cues[0].flagged)
```

## Safety

Advisory cues only. Humans decide evidence grading.
