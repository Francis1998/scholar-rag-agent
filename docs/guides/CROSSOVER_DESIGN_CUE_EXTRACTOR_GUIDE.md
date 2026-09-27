# CrossoverDesignCueExtractor Guide

![CrossoverDesignCueExtractor](../assets/crossover-design-cue-extractor.gif)

Offline evidence-methods cue extractor for crossover / Latin-square / ABAB
designs. Never network I/O. Gap vs Elicit / Consensus / CONSORT crossover.

Optional later polish via **GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2**.

Distinct from `ClusterRandomizationCueExtractor` and `WashoutPeriodCueExtractor`.

## Usage

```python
from retrieval.crossover_design_cues import CrossoverDesignCueExtractor

rows = CrossoverDesignCueExtractor().extract(
    [{"paper_id": "p1", "abstract": "A randomized crossover trial of therapy."}]
)
assert rows[0].flagged is True
```
