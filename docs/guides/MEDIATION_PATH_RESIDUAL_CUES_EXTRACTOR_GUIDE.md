# MediationPathResidualCueExtractor Guide

![MediationPathResidualCueExtractor](../../assets/demo/mediation-path-residual-cues.gif)

Offline helper.
Gap vs Elicit/Consensus/Connected Papers mediation-path residual cue extractors. Distinct from `MediationAnalysisCueExtractor` / `MediationBiasCueExtractor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.mediation_path_residual_cues import MediationPathResidualCueExtractor

rows = MediationPathResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss mediation path residual."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
