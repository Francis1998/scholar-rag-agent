# TransportabilityCueExtractor Guide

![TransportabilityCueExtractor](../assets/transportability-cue-extractor.gif)

Closes Elicit / Consensus / external-validity transportability gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `SubgroupAnalysisCueExtractor` and `ConfoundingAdjustmentCueExtractor`.

## Usage

```python
from retrieval.transportability_cues import TransportabilityCueExtractor

rows = TransportabilityCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "We assessed transportability to the target population for external validity.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
