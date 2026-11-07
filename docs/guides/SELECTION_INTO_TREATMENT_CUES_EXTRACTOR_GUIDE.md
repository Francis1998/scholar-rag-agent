# SelectionIntoTreatmentCueExtractor Guide

![SelectionIntoTreatmentCueExtractor](../../assets/demo/selection-into-treatment-cues.gif)

Offline cue extractor.
Gap vs Elicit/Consensus/Connected Papers selection-into-treatment cue extractors. Distinct from `SelectionBiasCueExtractor` / `HealthyWorkerSurvivorCueExtractor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.selection_into_treatment_cues import SelectionIntoTreatmentCueExtractor

rows = SelectionIntoTreatmentCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss selection into treatment."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O.
