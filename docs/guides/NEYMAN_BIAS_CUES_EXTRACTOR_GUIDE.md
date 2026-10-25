# NeymanBiasCueExtractor Guide

![NeymanBiasCueExtractor](../../assets/demo/neyman-bias-cues.gif)

Offline bias-cue extractor for literature triage.
Gap vs Elicit/Consensus/Connected Papers Neyman bias cue extractors. Distinct from `BerksonBiasCueExtractor` / `SelectionBiasCueExtractor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.neyman_bias_cues import NeymanBiasCueExtractor

rows = NeymanBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Evidence of Neyman bias from selective survival."}],
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory cues only. No network I/O. Humans decide inclusion.
