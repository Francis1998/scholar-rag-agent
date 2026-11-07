# MeasurementErrorBiasCueExtractor Guide

![MeasurementErrorBiasCueExtractor flow](../assets/measurement-error-bias-cue-extractor.gif)

Offline cue extractor. Never network I/O.

Gap vs Elicit/Consensus/AJE measurement-error bias cue extractors.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.measurement_error_bias_cues import MeasurementErrorBiasCueExtractor

cues = MeasurementErrorBiasCueExtractor().extract([{"paper_id": "p1", "abstract": "..."}])
print(cues[0].flagged)
```

## Safety

Advisory cues only. Humans decide evidence grading.
