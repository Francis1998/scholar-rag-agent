# MendelianRandomizationCueExtractor Guide

![MendelianRandomizationCueExtractor](../assets/mendelian-randomization-cue-extractor.gif)

Closes Elicit / Consensus / AJE Mendelian randomization gaps.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `TransportabilityCueExtractor` and `ConfoundingAdjustmentCueExtractor`.

## Usage

```python
from retrieval.mendelian_randomization_cues import MendelianRandomizationCueExtractor

rows = MendelianRandomizationCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "We performed Mendelian randomization with MR-Egger for pleiotropy.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
