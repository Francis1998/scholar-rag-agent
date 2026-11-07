# EcologicalFallacyCueExtractor Guide

![EcologicalFallacyCueExtractor](../assets/ecological-fallacy-cues.gif)

Offline evidence cue extractor. Never network I/O. Gap vs Elicit/Consensus/AJE ecological-fallacy cue extractors.

Optional later narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `SimpsonParadoxCueExtractor` / `TransportabilityCueExtractor`.

## Usage

```python
from retrieval.ecological_fallacy_cues import EcologicalFallacyCueExtractor

rows = [
    {"paper_id": "p1", "abstract": "We discuss ecological fallacy when interpreting group rates."}
]
out = EcologicalFallacyCueExtractor().extract(rows)
print(out[0].flagged, out[0].cue_kind)
```

## Safety

Advisory cues only. Humans decide.
