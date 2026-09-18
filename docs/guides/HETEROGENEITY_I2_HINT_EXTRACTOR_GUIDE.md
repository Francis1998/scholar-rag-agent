# Heterogeneity I2 Hint Extractor Guide

![Heterogeneity I2 hint extractor demo](../assets/heterogeneity-i2-hint-extractor.gif)

`HeterogeneityI2HintExtractor` scans title / abstract / results text for
offline meta-analysis heterogeneity cues (`I2 = 45%`, `I^2=60%`, unicode
superscript-2 forms, `substantial heterogeneity`, `between-study
heterogeneity`). It returns advisory `i2_percent` / `cue_kind` pairs with
the matched cue string. It never calls the network.

Distinct from `EffectSizeHintExtractor` (Cohen's d / OR / HR / RR / AUC),
`ConfidenceIntervalHintExtractor` (CI bounds), and `PValueHintExtractor`
(p-value cues). Fills an Elicit / Consensus / SciSpace / PaperQA
heterogeneity / I2 surfacing gap with deterministic heuristics. Optional
later narrative can use GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2.

## Usage

```python
from retrieval.heterogeneity_i2_hint import HeterogeneityI2HintExtractor

hints = HeterogeneityI2HintExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Pooled analysis showed I2 = 45% across trials.",
        },
        {
            "paper_id": "clear",
            "abstract": "We study transformers without meta-analytic pooling.",
        },
    ]
)
for hint in hints:
    print(hint.paper_id, hint.cue_kind, hint.i2_percent, hint.matched_cue)
```

Numeric I2 matches are preferred over qualitative heterogeneity phrases.
Empty paper lists return an empty tuple. Inputs are not mutated.
