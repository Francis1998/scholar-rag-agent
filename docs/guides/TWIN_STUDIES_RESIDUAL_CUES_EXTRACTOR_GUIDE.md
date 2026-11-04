# TwinStudiesResidualCueExtractor Guide

![TwinStudiesResidualCueExtractor](../../assets/demo/twin-studies-residual-cues.gif)

Offline helper.
Gap vs Elicit/Consensus/Connected Papers twin-studies residual cue extractors. Distinct from `MendelianRandomizationCueExtractor` / `MendelianRandomizationWeakCueExtractor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.twin_studies_residual_cues import TwinStudiesResidualCueExtractor

rows = TwinStudiesResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss twin studies residual."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
