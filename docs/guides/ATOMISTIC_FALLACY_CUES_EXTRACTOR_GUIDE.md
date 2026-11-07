# AtomisticFallacyCueExtractor Guide

![AtomisticFallacyCueExtractor](../../assets/demo/atomistic-fallacy-cues.gif)

Offline HITL helper for `cues` bands.
Gap vs Elicit/Consensus/Connected Papers atomistic-fallacy cue extractors. Distinct from `EcologicalFallacyCueExtractor` / `SimpsonParadoxCueExtractor`.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.atomistic_fallacy_cues import AtomisticFallacyCueExtractor

rows = AtomisticFallacyCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Authors note atomistic fallacy concerns.",
        }
    ],
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts. Humans decide.
