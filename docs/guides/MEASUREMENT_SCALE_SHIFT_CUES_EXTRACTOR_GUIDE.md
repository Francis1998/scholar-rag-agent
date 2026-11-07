# MeasurementScaleShiftCueExtractor Guide

![MeasurementScaleShiftCueExtractor](../../assets/demo/measurement-scale-shift-cues.gif)

Offline advisory bias cues.
Gap vs Elicit/Consensus/Connected Papers measurement-scale shift cue extractors. Distinct from `MeasurementErrorBiasCueExtractor` / `NonDifferentialMisclassificationCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.measurement_scale_shift_cues import MeasurementScaleShiftCueExtractor

rows = MeasurementScaleShiftCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Evidence of measurement-scale shift after instrument rescaling.",
        }
    ],
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
