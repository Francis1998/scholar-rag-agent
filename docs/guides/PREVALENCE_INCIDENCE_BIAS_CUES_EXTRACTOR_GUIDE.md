# PrevalenceIncidenceBiasCueExtractor Guide

![PrevalenceIncidenceBiasCueExtractor](../assets/prevalence-incidence-bias-cues.gif)

Offline cue extractor for deterministic title/abstract/methods patterns.
Gap vs Elicit/Consensus/Semantic Scholar prevalence-incidence bias cue extractors. Distinct from `SelectionBiasCueExtractor` / `ImmortalTimeBiasCueExtractor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

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
