# InterviewerBiasCueExtractor Guide

![InterviewerBiasCueExtractor flow](../assets/interviewer-bias-cue-extractor.gif)

Offline cue extractor. Never network I/O.

Gap vs Elicit/Consensus/AJE interviewer-bias cue extractors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.interviewer_bias_cues import InterviewerBiasCueExtractor

cues = InterviewerBiasCueExtractor().extract([{"paper_id": "p1", "abstract": "..."}])
print(cues[0].flagged)
```

## Safety

Advisory cues only. Humans decide evidence grading.
