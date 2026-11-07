# ChannelingBiasCueExtractor Guide

![ChannelingBiasCueExtractor](../assets/channeling-bias-cues.gif)

Offline deterministic cue extractor. Gap vs Elicit/Consensus/AJE channeling-bias cue extractors.
Distinct from `HealthyUserBiasCueExtractor` / `ResidualConfoundingCueExtractor`.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.channeling_bias_cues import ChannelingBiasCueExtractor

rows = ChannelingBiasCueExtractor().extract([{"paper_id": "p1", "abstract": "channeling bias"}])
print(rows[0].flagged)
```

## Safety

Advisory cues only. No network I/O.
