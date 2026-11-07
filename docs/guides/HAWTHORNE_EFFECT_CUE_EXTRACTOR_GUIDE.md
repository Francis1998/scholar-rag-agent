# HawthorneEffectCueExtractor Guide

![HawthorneEffectCueExtractor](../assets/hawthorne-effect-cues.gif)

Offline deterministic cue extractor. Gap vs Elicit/Consensus/Connected Papers Hawthorne-effect cue extractors.
Distinct from `HealthyUserBiasCueExtractor` / `InformationBiasCueExtractor`.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.hawthorne_effect_cues import HawthorneEffectCueExtractor

rows = HawthorneEffectCueExtractor().extract([{"paper_id": "p1", "abstract": "hawthorne effect"}])
print(rows[0].flagged)
```

## Safety

Advisory cues only. No network I/O.
