# HealthyWorkerResidualCueExtractor Guide

![HealthyWorkerResidualCueExtractor](../../assets/demo/healthy-worker-residual-cues.gif)

Offline residual cue extractor.
Gap vs Elicit/Consensus/Connected Papers healthy-worker residual cue extractors. Distinct from `HealthyWorkerEffectCueExtractor` / `HealthyWorkerSurvivorCueExtractor`.

Stack: **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.healthy_worker_residual_cues import HealthyWorkerResidualCueExtractor

rows = HealthyWorkerResidualCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss healthy worker residual."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
