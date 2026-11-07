# AttritionResidualCueExtractor Guide

![AttritionResidualCueExtractor](../../assets/demo/attrition-residual-cues.gif)

Offline helper.
Gap vs Elicit/Consensus/Connected Papers attrition residual cue extractors. Distinct from `LossToFollowupBiasCueExtractor` / `InformativeCensoringCueExtractor`.

Stack: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.attrition_residual_cues import AttritionResidualCueExtractor

rows = AttritionResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "attrition residual"}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
