# MediationBiasCueExtractor Guide

![MediationBiasCueExtractor](../../assets/demo/mediation-bias-cues.gif)

Offline deterministic cue extractor.
Gap vs Elicit/Consensus/Connected Papers mediation-bias cue extractors. Distinct from `MediationAnalysisCueExtractor` / `SelectionIntoTreatmentCueExtractor`.

Optional narrative polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.mediation_bias_cues import MediationBiasCueExtractor

rows = MediationBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss mediation bias."}]
)
print(rows[0].cues)
```

## Safety

Advisory only. No network I/O.
