# BayesianInterimPriorCueExtractor Guide

![BayesianInterimPriorCueExtractor](../assets/bayesian-interim-prior-cue-extractor.gif)

Offline advisory cue extractor. Never calls the network.
Closes Elicit / Consensus / FDA Bayesian interim prior gaps.

Optional polish via **GPT-6 Astra / Claude Opus 5.5 / Gemini 4 Argon / Kimi K3.1 Pro**.

Distinct from InterimAnalysisCueExtractor / AdaptiveDesignCueExtractor.

## Usage

```python
from retrieval.bayesian_interim_prior_cues import BayesianInterimPriorCueExtractor

rows = BayesianInterimPriorCueExtractor().extract(
    [
        {
            "paper_id": "p1",
            "methods": "A weakly informative prior guided the Bayesian interim analysis.",
        }
    ]
)
print(rows[0].flagged, rows[0].cues)
```
