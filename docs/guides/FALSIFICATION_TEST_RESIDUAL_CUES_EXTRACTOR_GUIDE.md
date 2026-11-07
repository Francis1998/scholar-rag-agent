# FalsificationTestResidualCueExtractor Guide

![FalsificationTestResidualCueExtractor](../../assets/demo/falsification-test-residual-cues.gif)

Offline helper.
Gap vs Elicit/Consensus/Connected Papers falsification-test residual cue extractors. Distinct from `NegativeControlOutcomeCueExtractor` / `PlaceboTestCueExtractor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.falsification_test_residual_cues import FalsificationTestResidualCueExtractor

rows = FalsificationTestResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss falsification test residual."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
