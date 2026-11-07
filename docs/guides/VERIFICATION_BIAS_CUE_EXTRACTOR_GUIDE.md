# VerificationBiasCueExtractor Guide

![VerificationBiasCueExtractor demo](../assets/verification-bias-cues.gif)

Offline deterministic cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE verification-bias cue extractors.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.verification_bias_cues import VerificationBiasCueExtractor

cues = VerificationBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Results may reflect verification bias in tertiary clinics."}]
)
print(cues[0].flagged, cues[0].cue_kind)
```
