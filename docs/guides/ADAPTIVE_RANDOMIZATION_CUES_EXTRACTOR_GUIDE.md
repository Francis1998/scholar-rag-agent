# AdaptiveRandomizationCueExtractor Guide

![AdaptiveRandomizationCueExtractor](../../assets/demo/adaptive-randomization-cues.gif)

Offline cue extractor.
Gap vs Elicit/Consensus/Connected Papers adaptive-randomization cue extractors. Distinct from `AdaptiveDesignCueExtractor` / `AdaptiveEnrichmentCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.adaptive_randomization_cues import AdaptiveRandomizationCueExtractor

rows = AdaptiveRandomizationCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss adaptive randomization."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O.
