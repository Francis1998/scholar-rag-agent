# ResidualConfoundingCueExtractor Guide

![ResidualConfoundingCueExtractor](../assets/residual-confounding-cues.gif)

Offline cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE residual-confounding cue extractors.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ConfoundingByIndicationCueExtractor` / `TimeVaryingConfoundingCueExtractor`.

## Usage

```python
from retrieval.residual_confounding_cues import ResidualConfoundingCueExtractor

rows = [{"paper_id": "p1", "abstract": "We discuss residual confounding after adjustment."}]
print(ResidualConfoundingCueExtractor().extract(rows)[0].flagged)
```

## Safety

Advisory cues only. Humans decide.
