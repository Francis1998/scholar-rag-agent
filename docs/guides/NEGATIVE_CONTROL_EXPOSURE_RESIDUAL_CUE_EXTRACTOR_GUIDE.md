# NegativeControlExposureResidualCueExtractor Guide

![NegativeControlExposureResidualCueExtractor](../../assets/demo/negative-control-exposure-residual-cues.gif)

Offline residual cue extractor.
Gap vs Elicit/Consensus/Connected Papers negative-control exposure residual cue extractors. Distinct from `NegativeControlOutcomeCueExtractor` / `ResidualConfoundingCueExtractor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.negative_control_exposure_residual_cues import (
    NegativeControlExposureResidualCueExtractor,
)

rows = NegativeControlExposureResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss negative control exposure residual."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
