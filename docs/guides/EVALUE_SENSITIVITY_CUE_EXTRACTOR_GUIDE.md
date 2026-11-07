# EValueSensitivityCueExtractor Guide

![EValueSensitivityCueExtractor flow](../assets/evalue-sensitivity-cue-extractor.gif)

Offline cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE E-value sensitivity cue extractors.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ConfoundingAdjustmentCueExtractor / NegativeControlOutcomeCueExtractor`.

## Usage

```python
from retrieval.evalue_sensitivity_cues import EValueSensitivityCueExtractor

cues = EValueSensitivityCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "We report an E-value sensitivity analysis for unmeasured confounding.",
        }
    ]
)
assert cues[0].flagged is True
```
