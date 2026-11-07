# LateLookerBiasCueExtractor Guide

![LateLookerBiasCueExtractor](../../assets/demo/late-looker-bias-cues.gif)

Offline advisory cue extractor.
Gap vs Elicit/Consensus/Connected Papers late-looker bias cue extractors. Distinct from `LengthTimeBiasCueExtractor` / `LeadTimeBiasCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.late_looker_bias_cues import LateLookerBiasCueExtractor

rows = LateLookerBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Evidence of late-looker bias in the cohort."}],
)
assert rows[0].flagged is True
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
