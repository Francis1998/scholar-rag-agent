# EValueResidualCueExtractor Guide

![EValueResidualCueExtractor](../../assets/demo/evalue-residual-cues.gif)

Offline residual cue extractor.
Gap vs Elicit/Consensus/Connected Papers E-value residual cue extractors. Distinct from `EValueSensitivityCueExtractor` / `SelectionBiasResidualCueExtractor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.evalue_residual_cues import EValueResidualCueExtractor

rows = EValueResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss evalue residual."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
