# HawthorneEffectCueExtractor Guide

![HawthorneEffectCueExtractor](../assets/hawthorne-effect-cues.gif)

Offline deterministic cue extractor. Gap vs Elicit/Consensus/Connected Papers Hawthorne-effect cue extractors.
Distinct from `HealthyUserBiasCueExtractor` / `InformationBiasCueExtractor`.

Optional later narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.hawthorne_effect_cues import HawthorneEffectCueExtractor

rows = HawthorneEffectCueExtractor().extract([{"paper_id": "p1", "abstract": "hawthorne effect"}])
print(rows[0].flagged)
```

## Safety

Advisory cues only. No network I/O.
