# NoveltyBiasCueExtractor Guide

![NoveltyBiasCueExtractor](../assets/novelty-bias-cues.gif)

Offline deterministic cue extractor. Gap vs Elicit/Consensus/Semantic Scholar novelty-bias cue extractors.
Distinct from `InformationBiasCueExtractor` / `RightCensoringCueExtractor`.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.novelty_bias_cues import NoveltyBiasCueExtractor

rows = NoveltyBiasCueExtractor().extract([{"paper_id": "p1", "abstract": "novelty bias"}])
print(rows[0].flagged)
```

## Safety

Advisory cues only. No network I/O.
