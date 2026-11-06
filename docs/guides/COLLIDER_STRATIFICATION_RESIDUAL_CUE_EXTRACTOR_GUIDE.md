# ColliderStratificationResidualCueExtractor Guide

![ColliderStratificationResidualCueExtractor](../../assets/demo/collider-stratification-residual-cues.gif)

Offline residual cue extractor.
Gap vs Elicit/Consensus/Connected Papers collider-stratification residual cue extractors. Distinct from `ColliderStratificationCueExtractor` / `ColliderBiasInteractionCueExtractor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.collider_stratification_residual_cues import (
    ColliderStratificationResidualCueExtractor,
)

rows = ColliderStratificationResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss collider stratification residual."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
