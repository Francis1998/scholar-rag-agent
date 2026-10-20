# IndexEventBiasCueExtractor Guide

![IndexEventBiasCueExtractor](../assets/index-event-bias-cues.gif)

Offline cue extractor for scholarly RAG pipelines.
Gap vs Elicit/Consensus/Connected Papers index-event bias cue extractors. Distinct from `BerksonBiasCueExtractor` / `SelectionBiasCueExtractor`.

Optional polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.index_event_bias_cues import IndexEventBiasCueExtractor

rows = IndexEventBiasCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "Mentions relevant bias cues."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory cues only. No network I/O. Humans decide.
