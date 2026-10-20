# NonDifferentialMisclassificationCueExtractor Guide

![NonDifferentialMisclassificationCueExtractor](../assets/non-differential-misclassification-cues.gif)

Offline cue extractor for scholarly RAG pipelines.
Gap vs Elicit/Consensus/AJE non-differential misclassification cue extractors. Distinct from `MeasurementErrorBiasCueExtractor` / `InformationBiasCueExtractor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.non_differential_misclassification_cues import NonDifferentialMisclassificationCueExtractor

rows = NonDifferentialMisclassificationCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Mentions relevant bias cues."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory cues only. No network I/O. Humans decide.
