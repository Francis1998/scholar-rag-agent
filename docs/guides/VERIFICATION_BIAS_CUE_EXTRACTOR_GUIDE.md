# VerificationBiasCueExtractor Guide

![VerificationBiasCueExtractor demo](../assets/verification-bias-cues.gif)

Offline deterministic cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE verification-bias cue extractors.

Optional later narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.verification_bias_cues import VerificationBiasCueExtractor

cues = VerificationBiasCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Diagnostic accuracy may be distorted by verification bias when only positives are imaged.",
        }
    ]
)
print(cues[0].flagged, cues[0].cue_kind)
```
