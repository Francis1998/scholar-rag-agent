# TimeDependentBiasCueExtractor Guide

![TimeDependentBiasCueExtractor](../../assets/demo/time-dependent-bias-cues.gif)

Offline cue extractor for time dependent bias signals.
Gap vs Elicit/Consensus/Connected Papers time-dependent bias cue extractors. Distinct from `ImmortalTimeBiasCueExtractor` / `ProtopathicBiasCueExtractor`.

Optional narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

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
