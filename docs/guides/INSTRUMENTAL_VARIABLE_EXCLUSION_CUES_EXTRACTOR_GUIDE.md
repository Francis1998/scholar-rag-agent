# InstrumentalVariableExclusionCueExtractor Guide

![InstrumentalVariableExclusionCueExtractor](../../assets/demo/instrumental-variable-exclusion-cues.gif)

Offline cue extractor.
Gap vs Elicit/Consensus/Connected Papers instrumental-variable exclusion cue extractors. Distinct from `InstrumentalVariableStrengthCueExtractor` / `InstrumentStrengthCueExtractor`.

Optional narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.instrumental_variable_exclusion_cues import InstrumentalVariableExclusionCueExtractor

rows = InstrumentalVariableExclusionCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "We discuss instrumental variable exclusion."}]
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O.
