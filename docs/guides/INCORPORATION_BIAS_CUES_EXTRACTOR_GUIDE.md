# IncorporationBiasCueExtractor Guide

![IncorporationBiasCueExtractor](../assets/incorporation-bias-cues.gif)

Offline cue extractor for deterministic title/abstract/methods patterns.
Gap vs Elicit/Consensus/AJE incorporation-bias cue extractors. Distinct from `VerificationBiasCueExtractor` / `SpectrumBiasCueExtractor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.incorporation_bias_cues import IncorporationBiasCueExtractor

rows = IncorporationBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Methods note incorporation bias from reference standard."}]
)
assert rows[0].flagged is True
```

## Safety

Advisory only. No network I/O. Humans decide.
