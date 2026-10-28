# TargetTrialEmulationCueExtractor Guide

![TargetTrialEmulationCueExtractor](../../assets/demo/target-trial-emulation-cues.gif)

Offline advisory cue extractor.
Gap vs Elicit/Consensus/Connected Papers target-trial emulation cue extractors. Distinct from `TimeDependentBiasCueExtractor` / `ImmortalTimeBiasCueExtractor`.

Optional narrative via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

## Usage

```python
from retrieval.target_trial_emulation_cues import TargetTrialEmulationCueExtractor

rows = TargetTrialEmulationCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "abstract": "Target Trial Emulation may clarify causal contrasts in observational cohorts.",
        }
    ],
)
print(rows[0].flagged, rows[0].cues)
```

## Safety

Advisory only. No network I/O. Never auto-acts.
