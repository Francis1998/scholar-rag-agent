# SurveyModeEffectCueExtractor Guide

![SurveyModeEffectCueExtractor](../../assets/demo/survey-mode-effect-cues.gif)

Offline advisory cue extractor.
Gap vs Elicit/Consensus/Connected Papers survey-mode effect cue extractors. Distinct from `MeasurementScaleShiftCueExtractor` / `MeasurementErrorBiasCueExtractor`.

Optional narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.survey_mode_effect_cues import SurveyModeEffectCueExtractor

rows = SurveyModeEffectCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "survey mode effect"}],
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
