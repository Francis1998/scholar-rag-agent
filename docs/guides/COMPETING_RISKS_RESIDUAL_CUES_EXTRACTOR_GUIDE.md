# CompetingRisksResidualCueExtractor Guide

![CompetingRisksResidualCueExtractor](../../assets/demo/competing-risks-residual-cues.gif)

Offline deterministic cue extractor.
Gap vs Elicit/Consensus/Connected Papers competing-risks residual cue extractors. Distinct from `CompetingRiskCueExtractor` / `CompetingRiskInformativeCensoringCueExtractor`.

Optional narrative polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

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
