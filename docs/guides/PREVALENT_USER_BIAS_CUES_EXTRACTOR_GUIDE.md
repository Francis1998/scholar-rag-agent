# PrevalentUserBiasCueExtractor Guide

![PrevalentUserBiasCueExtractor](../../assets/demo/prevalent-user-bias-cues.gif)

Offline cue extractor for prevalent user bias signals.
Gap vs Elicit/Consensus/Connected Papers prevalent-user bias cue extractors. Distinct from `HealthyUserBiasCueExtractor` / `DepletionOfSusceptiblesCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.prevalent_user_bias_cues import PrevalentUserBiasCueExtractor

rows = PrevalentUserBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Evidence of prevalent user bias in the cohort."}],
)
assert rows[0].flagged is True
```

## Safety

Advisory only. No network I/O. Never auto-acts.
