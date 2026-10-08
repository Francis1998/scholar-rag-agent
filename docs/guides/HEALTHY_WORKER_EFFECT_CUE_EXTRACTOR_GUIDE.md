# HealthyWorkerEffectCueExtractor Guide

![HealthyWorkerEffectCueExtractor flow](../assets/healthy-worker-effect-cue-extractor.gif)

Offline cue extractor. Never network I/O.

Gap vs Elicit/Consensus/AJE healthy-worker-effect cue extractors.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.healthy_worker_effect_cues import HealthyWorkerEffectCueExtractor

cues = HealthyWorkerEffectCueExtractor().extract([{"paper_id": "p1", "abstract": "..."}])
print(cues[0].flagged)
```

## Safety

Advisory cues only. Humans decide evidence grading.
