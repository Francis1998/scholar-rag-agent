# ReferralBiasCueExtractor Guide

![ReferralBiasCueExtractor](../assets/referral-bias-cues.gif)

Offline cue extractor for deterministic title/abstract/methods patterns.
Gap vs Elicit/Consensus/Connected Papers referral-bias cue extractors. Distinct from `ChannelingBiasCueExtractor` / `AscertainmentBiasCueExtractor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.referral_bias_cues import ReferralBiasCueExtractor

rows = ReferralBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Authors note referral bias in clinic enrollment."}]
)
assert rows[0].flagged is True
```

## Safety

Advisory only. No network I/O. Humans decide.
