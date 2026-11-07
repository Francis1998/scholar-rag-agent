# Absolute Risk Reduction Hint Extractor Guide

![ARR hint extractor demo](../assets/arr-hint-extractor.gif)

`AbsoluteRiskReductionHintExtractor` extracts offline ARR / absolute risk
reduction / risk difference numeric hints. Never calls the network.

Distinct from `NumberNeededToTreatHintExtractor`, `EffectSizeHintExtractor`,
and `PValueHintExtractor`. Fills an Elicit / Consensus / SciSpace / PaperQA
ARR gap. Optional later narrative: **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon /
Kimi K3.1 Pro**.

## Usage

```python
from retrieval.arr_hint import AbsoluteRiskReductionHintExtractor

rows = AbsoluteRiskReductionHintExtractor().extract(
    [{"paper_id": "p1", "abstract": "ARR = 3.2% favoring treatment."}]
)
print(rows[0].value, rows[0].unit)
```
