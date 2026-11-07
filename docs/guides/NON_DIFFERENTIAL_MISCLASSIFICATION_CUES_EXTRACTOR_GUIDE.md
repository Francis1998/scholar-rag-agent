# NonDifferentialMisclassificationCueExtractor Guide

![NonDifferentialMisclassificationCueExtractor](../assets/non-differential-misclassification-cues.gif)

Offline cue extractor for scholarly RAG pipelines.
Gap vs Elicit/Consensus/AJE non-differential misclassification cue extractors. Distinct from `MeasurementErrorBiasCueExtractor` / `InformationBiasCueExtractor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.non_differential_misclassification_cues import (
    NonDifferentialMisclassificationCueExtractor,
)

rows = NonDifferentialMisclassificationCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Mentions relevant bias cues."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory cues only. No network I/O. Humans decide.
