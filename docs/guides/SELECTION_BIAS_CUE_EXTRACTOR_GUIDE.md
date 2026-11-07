# SelectionBiasCueExtractor Guide

![SelectionBiasCueExtractor flow](../assets/selection-bias-cue-extractor.gif)

Offline cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE selection-bias cue extractors.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `EValueSensitivityCueExtractor / SpilloverInterferenceCueExtractor`.

## Usage

```python
from retrieval.selection_bias_cues import SelectionBiasCueExtractor

cues = SelectionBiasCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "We discuss selection bias, healthy-user bias, and volunteer bias.",
        }
    ]
)
assert cues[0].flagged is True
```
