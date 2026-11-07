# AscertainmentDifferentialResidualCueExtractor Guide

![AscertainmentDifferentialResidualCueExtractor](../../assets/demo/ascertainment-differential-residual-cues.gif)

Offline advisory cue extractor.
Gap vs Elicit/Consensus/Connected Papers ascertainment-differential residual cue extractors. Distinct from `AscertainmentBiasCueExtractor` / `SelectionBiasCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.ascertainment_differential_residual_cues import (
    AscertainmentDifferentialResidualCueExtractor,
)

rows = AscertainmentDifferentialResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "ascertainment differential residual"}],
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
