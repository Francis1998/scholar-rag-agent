# MediationAnalysisCueExtractor Guide

![MediationAnalysisCueExtractor](../assets/mediation-analysis-cue-extractor.gif)

Closes Elicit / Consensus / AJE mediation-analysis gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ConfoundingAdjustmentCueExtractor` and `IntentionToTreatCueExtractor`.

## Usage

```python
from retrieval.mediation_analysis_cues import MediationAnalysisCueExtractor

rows = MediationAnalysisCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "We conducted a causal mediation analysis of the indirect effect.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
