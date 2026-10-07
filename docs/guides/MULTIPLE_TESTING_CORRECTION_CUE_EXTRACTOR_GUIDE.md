# MultipleTestingCorrectionCueExtractor Guide

![MultipleTestingCorrectionCueExtractor flow](../assets/multiple-testing-correction-cue-extractor.gif)

Offline cue extractor. Never network I/O.

Gap vs Elicit/Consensus/AJE multiple-testing correction cue extractors.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.multiple_testing_correction_cues import MultipleTestingCorrectionCueExtractor

cues = MultipleTestingCorrectionCueExtractor().extract([{"paper_id": "p1", "abstract": "..."}])
print(cues[0].flagged)
```

## Safety

Advisory cues only. Humans decide evidence grading.
