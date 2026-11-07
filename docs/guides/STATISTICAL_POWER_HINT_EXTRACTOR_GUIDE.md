# Statistical Power Hint Extractor Guide

![Statistical power hint extractor demo](../assets/statistical-power-hint-extractor.gif)

`StatisticalPowerHintExtractor` extracts offline power-percent and power-
calculation cues. Never calls the network.

Distinct from `SampleSizeHintExtractor`, `PValueHintExtractor`, and
`ConfidenceIntervalHintExtractor`. Fills an Elicit / Consensus / SciSpace /
PaperQA power-analysis gap. Optional later narrative: **GPT-6 Astra / Claude
Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

## Usage

```python
from retrieval.power_hint import StatisticalPowerHintExtractor

rows = StatisticalPowerHintExtractor().extract(
    [{"paper_id": "p1", "methods": "The study had 80% power."}]
)
print(rows[0].power_percent, rows[0].cues)
```
