# PrincipalStratumResidualCueExtractor Guide

![PrincipalStratumResidualCueExtractor](../../assets/demo/principal-stratum-residual-cues.gif)

Offline helper.
Gap vs Elicit/Consensus/Connected Papers principal-stratum residual cue extractors. Distinct from `InstrumentalVariableExclusionCueExtractor` / `CrossoverDesignCueExtractor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.principal_stratum_residual_cues import PrincipalStratumResidualCueExtractor

rows = PrincipalStratumResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss principal stratum residual."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
