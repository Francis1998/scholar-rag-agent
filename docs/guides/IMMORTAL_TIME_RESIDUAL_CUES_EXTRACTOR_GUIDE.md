# ImmortalTimeResidualCueExtractor Guide

![ImmortalTimeResidualCueExtractor](../../assets/demo/immortal-time-residual-cues.gif)

Offline cue extractor.
Gap vs Elicit/Consensus/Connected Papers immortal-time residual cue extractors. Distinct from `SurvivorBiasCueExtractor` / `TimeDependentBiasCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.immortal_time_residual_cues import ImmortalTimeResidualCueExtractor

rows = ImmortalTimeResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss immortal time residual in the cohort."}]
)
print(rows[0].cues)
```

## Safety

Advisory only. No network I/O.
