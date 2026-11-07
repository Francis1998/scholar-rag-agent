# DifferentialMisclassificationCueExtractor Guide

![DifferentialMisclassificationCueExtractor](../../assets/demo/differential-misclassification-cues.gif)

Offline cue extractor for scholarly evidence screening.
Gap vs Elicit/Consensus/AJE differential misclassification cue extractors. Distinct from `NonDifferentialMisclassificationCueExtractor` / `MeasurementErrorBiasCueExtractor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.differential_misclassification_cues import DifferentialMisclassificationCueExtractor

rows = DifferentialMisclassificationCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Study notes differential misclassification in methods."}],
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory cues only. No network I/O. Humans decide.
