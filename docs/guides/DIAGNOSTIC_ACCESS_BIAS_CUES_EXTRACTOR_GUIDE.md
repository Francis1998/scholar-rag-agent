# DiagnosticAccessBiasCueExtractor Guide

![DiagnosticAccessBiasCueExtractor](../assets/diagnostic-access-bias-cues.gif)

Offline cue extractor for deterministic title/abstract/methods patterns.
Gap vs Elicit/Consensus/Connected Papers diagnostic-access bias cue extractors. Distinct from `DetectionBiasCueExtractor` / `VerificationBiasCueExtractor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.diagnostic_access_bias_cues import DiagnosticAccessBiasCueExtractor

rows = DiagnosticAccessBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Study discusses diagnostic access bias."}]
)
assert rows[0].flagged is True
```

## Safety

Advisory only. No network I/O. Humans decide.
