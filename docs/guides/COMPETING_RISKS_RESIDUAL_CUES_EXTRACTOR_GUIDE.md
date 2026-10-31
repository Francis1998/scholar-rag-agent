# CompetingRisksResidualCueExtractor Guide

![CompetingRisksResidualCueExtractor](../../assets/demo/competing-risks-residual-cues.gif)

Offline deterministic cue extractor.
Gap vs Elicit/Consensus/Connected Papers competing-risks residual cue extractors. Distinct from `CompetingRiskCueExtractor` / `CompetingRiskInformativeCensoringCueExtractor`.

Optional narrative polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.competing_risks_residual_cues import CompetingRisksResidualCueExtractor

rows = CompetingRisksResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss competing risks residual."}]
)
print(rows[0].cues)
```

## Safety

Advisory only. No network I/O.
