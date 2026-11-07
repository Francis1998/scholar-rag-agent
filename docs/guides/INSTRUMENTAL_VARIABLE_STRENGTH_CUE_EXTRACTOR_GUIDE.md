# InstrumentalVariableStrengthCueExtractor Guide

![InstrumentalVariableStrengthCueExtractor](../assets/instrumental-variable-strength-cue-extractor.gif)

Closes Elicit / Consensus / AJE instrumental-variable strength gaps.

Optional later polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from `ConfoundingAdjustmentCueExtractor` and `TransportabilityCueExtractor`.

## Usage

```python
from retrieval.instrumental_variable_strength_cues import InstrumentalVariableStrengthCueExtractor

rows = InstrumentalVariableStrengthCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "The first-stage F-statistic suggested a weak instrument.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Offline patterns only. No HTTP. Humans interpret evidence cues.
