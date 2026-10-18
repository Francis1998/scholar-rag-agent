# PrevalenceIncidenceBiasCueExtractor Guide

![PrevalenceIncidenceBiasCueExtractor](../assets/prevalence-incidence-bias-cues.gif)

Offline cue extractor for deterministic title/abstract/methods patterns.
Gap vs Elicit/Consensus/Semantic Scholar prevalence-incidence bias cue extractors. Distinct from `SelectionBiasCueExtractor` / `ImmortalTimeBiasCueExtractor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.prevalence_incidence_bias_cues import PrevalenceIncidenceBiasCueExtractor

rows = PrevalenceIncidenceBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Study discusses prevalence-incidence bias in the cohort."}]
)
assert rows[0].flagged is True
```

## Safety

Advisory only. No network I/O. Humans decide.
