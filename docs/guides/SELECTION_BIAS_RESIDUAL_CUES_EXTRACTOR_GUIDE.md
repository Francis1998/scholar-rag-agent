# SelectionBiasResidualCueExtractor Guide

![SelectionBiasResidualCueExtractor](../../assets/demo/selection-bias-residual-cues.gif)

Offline helper.
Gap vs Elicit/Consensus/Connected Papers selection-bias residual cue extractors. Distinct from `SelectionBiasCueExtractor` / `SelectionIntoTreatmentCueExtractor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.selection_bias_residual_cues import SelectionBiasResidualCueExtractor

rows = SelectionBiasResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "selection bias residual"}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
