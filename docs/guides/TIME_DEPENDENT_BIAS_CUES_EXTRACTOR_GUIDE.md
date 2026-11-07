# TimeDependentBiasCueExtractor Guide

![TimeDependentBiasCueExtractor](../../assets/demo/time-dependent-bias-cues.gif)

Offline cue extractor for time dependent bias signals.
Gap vs Elicit/Consensus/Connected Papers time-dependent bias cue extractors. Distinct from `ImmortalTimeBiasCueExtractor` / `ProtopathicBiasCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.time_dependent_bias_cues import TimeDependentBiasCueExtractor

rows = TimeDependentBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Evidence of time dependent bias in the cohort."}],
)
assert rows[0].flagged is True
```

## Safety

Advisory only. No network I/O. Never auto-acts.
