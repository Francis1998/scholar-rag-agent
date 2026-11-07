# SurveyModeEffectCueExtractor Guide

![SurveyModeEffectCueExtractor](../../assets/demo/survey-mode-effect-cues.gif)

Offline advisory cue extractor.
Gap vs Elicit/Consensus/Connected Papers survey-mode effect cue extractors. Distinct from `MeasurementScaleShiftCueExtractor` / `MeasurementErrorBiasCueExtractor`.

Optional narrative via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

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
