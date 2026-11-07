# ColliderBiasOpeningCueExtractor Guide

![ColliderBiasOpeningCueExtractor](../../assets/demo/collider-bias-opening-cues.gif)

Offline cue extractor.
Gap vs Elicit/Consensus/Connected Papers collider-bias-opening cue extractors. Distinct from `SelectionBiasCueExtractor` / `SimpsonParadoxCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.collider_bias_opening_cues import ColliderBiasOpeningCueExtractor

rows = ColliderBiasOpeningCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss collider bias opening in the cohort."}]
)
print(rows[0].cues)
```

## Safety

Advisory only. No network I/O.
