# NonInferiorityMarginCueExtractor Guide

![NonInferiorityMarginCueExtractor](../assets/noninferiority-margin-cue-extractor.gif)

Offline evidence-methods cue extractor for non-inferiority / equivalence margins.
Never network I/O. Gap vs Elicit / Consensus / CONSORT / FDA.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `EffectSizeHintExtractor` and `PrimaryEndpointCueExtractor`.

## Usage

```python
from retrieval.noninferiority_margin_cues import NonInferiorityMarginCueExtractor

rows = NonInferiorityMarginCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "This was a non-inferiority trial versus SOC."}]
)
assert rows[0].flagged is True
```
